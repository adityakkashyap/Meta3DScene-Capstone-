#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import psycopg2
from sentence_transformers import SentenceTransformer
import json
import re
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize embedding model (only once)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432))
}

def store_generation_result(
    prompt: str,
    success: bool,
    project_path: str,
    generation_time_seconds: float,
    correction_attempts: int,
    final_attempt_number: int,
    clip_scene_score: float = None,
    clip_overhead_score: float = None,
    clip_ground_score: float = None,
    validation_passed: bool = None,
    blender_metrics: dict = None,
    assets_requested: int = 0,
    assets_found: int = 0,
    assets_used: int = 0,
    final_error_log: str = None,
    error_type: str = None
):
    """
    Stores main generation result to database
    Returns generation_id for linking child records
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Generate embedding for semantic search
    embedding = embedding_model.encode(prompt)  # numpy array
    # format as pgvector literal string: [v1,v2,...]
    prompt_embedding_str = "[" + ",".join(f"{float(v):.18g}" for v in embedding) + "]"

    # Extract Blender metrics if available
    terrain_relief_m = blender_metrics.get('terrain_relief_m') if blender_metrics else None
    terrain_rms_disp = blender_metrics.get('terrain_rms_disp') if blender_metrics else None
    mean_nnd_xy = blender_metrics.get('mean_nnd_xy') if blender_metrics else None
    global_density = blender_metrics.get('global_density_per_m2') if blender_metrics else None
    out_of_bounds = blender_metrics.get('out_of_bounds_count') if blender_metrics else None
    coverage_pct = blender_metrics.get('projected_coverage_pct') if blender_metrics else None
    total_verts = blender_metrics.get('total_vertices') if blender_metrics else None
    total_tris = blender_metrics.get('total_triangles') if blender_metrics else None

    cur.execute("""
        INSERT INTO generations (
            prompt, prompt_embedding, project_path, 
            generation_time_seconds, success, correction_attempts, final_attempt_number,
            clip_scene_score, clip_overhead_score, clip_ground_score, validation_passed,
            terrain_relief_m, terrain_rms_disp, mean_nnd_xy, global_density_per_m2,
            out_of_bounds_count, projected_coverage_pct, total_vertices, total_triangles,
            assets_requested, assets_found, assets_used,
            final_error_log, error_type
        ) VALUES (
            %s, %s::vector, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) RETURNING id
    """, (
        prompt, prompt_embedding_str, project_path,
        generation_time_seconds, success, correction_attempts, final_attempt_number,
        clip_scene_score, clip_overhead_score, clip_ground_score, validation_passed,
        terrain_relief_m, terrain_rms_disp, mean_nnd_xy, global_density,
        out_of_bounds, coverage_pct, total_verts, total_tris,
        assets_requested, assets_found, assets_used,
        final_error_log, error_type
    ))

    generation_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    print(f"✅ Stored generation record (ID: {generation_id})")
    return generation_id


def store_asset_details(generation_id: int, measured_assets_result: dict, detailed_plan: list):
    """
    Stores per-asset details including CLIP scores and usage stats
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    assets_mapping = measured_assets_result.get('assets_mapping', {})
    clip_thumbnails = measured_assets_result.get('clip_thumbnails', {})

    # ✅ FIX: Better handling of detailed_plan structure
    target_heights = {}
    instance_counts = {}

    if isinstance(detailed_plan, list):
        for item in detailed_plan:
            # Handle both 'name' and 'asset_name' keys
            name = item.get('name') or item.get('asset_name')
            if name:
                target_heights[name] = item.get('target_height_meters') or item.get('target_height')
                instance_counts[name] = item.get('count') or item.get('instance_count')

    for asset_name, asset_info in assets_mapping.items():
        # Get measured dimensions
        measured_dims = asset_info.get('measured_dimensions', {})
        measured_x = measured_dims.get('x')
        measured_y = measured_dims.get('y')
        measured_z = measured_dims.get('z')

        # Get CLIP thumbnail scores
        clip_data = clip_thumbnails.get(asset_name, {})
        clip_score = clip_data.get('clip_score')
        clip_best_prompt = clip_data.get('best_prompt')
        thumbnail_path = clip_data.get('thumbnail')

        # Get usage stats from detailed plan
        target_height = target_heights.get(asset_name)
        instance_count = instance_counts.get(asset_name)

        # ✅ DEBUG: Print what we're storing
        print(f"  Storing asset: {asset_name}")
        print(f"    - target_height: {target_height}")
        print(f"    - instance_count: {instance_count}")

        cur.execute("""
            INSERT INTO assets (
                generation_id, asset_name, sketchfab_uid, local_path,
                measured_x, measured_y, measured_z,
                clip_thumbnail_score, clip_best_prompt, thumbnail_path,
                target_height, instance_count
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            generation_id, asset_name, asset_info.get('uid'), asset_info.get('local_path'),
            measured_x, measured_y, measured_z,
            clip_score, clip_best_prompt, thumbnail_path,
            target_height, instance_count
        ))

    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Stored {len(assets_mapping)} asset records")


def store_correction_attempt(
    generation_id: int,
    attempt_number: int,
    agent_used: str,
    error_occurred: bool,
    error_traceback: str = None,
    execution_time_seconds: float = None
):
    """
    Stores each correction attempt with error details
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Parse error type and message from traceback
    error_type = None
    error_message = None
    error_line_number = None

    if error_traceback:
        # Extract error type (e.g., "SyntaxError", "AttributeError")
        error_type_match = re.search(r'(\w+Error):', error_traceback)
        if error_type_match:
            error_type = error_type_match.group(1)

        # Extract first line of error message
        lines = error_traceback.split('\n')
        for line in lines:
            if 'Error:' in line:
                error_message = line.strip()
                break

        # Extract line number
        line_match = re.search(r'line (\d+)', error_traceback)
        if line_match:
            error_line_number = int(line_match.group(1))

    cur.execute("""
        INSERT INTO correction_attempts (
            generation_id, attempt_number, agent_used, error_occurred,
            error_type, error_message, error_line_number, full_traceback,
            execution_time_seconds
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        generation_id, attempt_number, agent_used, error_occurred,
        error_type, error_message, error_line_number, error_traceback,
        execution_time_seconds
    ))

    conn.commit()
    cur.close()
    conn.close()

    status = "❌ ERROR" if error_occurred else "✅ SUCCESS"
    print(f"{status} Stored attempt #{attempt_number} ({agent_used})")


def update_asset_performance(asset_name: str, success: bool, clip_score: float = None):
    """
    Updates aggregate asset performance stats
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Check if asset exists
    cur.execute("SELECT id FROM asset_performance WHERE asset_name = %s", (asset_name,))
    exists = cur.fetchone()

    if exists:
        # Update existing record
        cur.execute("""
            UPDATE asset_performance SET
                times_used = times_used + 1,
                times_successful = times_successful + CASE WHEN %s THEN 1 ELSE 0 END,
                success_rate = (times_successful + CASE WHEN %s THEN 1 ELSE 0 END)::FLOAT / (times_used + 1),
                last_used = CURRENT_TIMESTAMP
            WHERE asset_name = %s
        """, (success, success, asset_name))
    else:
        # Insert new record
        cur.execute("""
            INSERT INTO asset_performance (
                asset_name, times_used, times_successful, success_rate
            ) VALUES (%s, 1, %s, %s)
        """, (asset_name, 1 if success else 0, 1.0 if success else 0.0))

    conn.commit()
    cur.close()
    conn.close()

print("✅ Storage functions loaded successfully!")


# In[ ]:


def validate_scene_with_clip(prompt, scene_folder, clip_threshold=0.25):
    """
    Validates scene-prompt matching using CLIP scores with dual-view rendering
    """
    import os
    import subprocess
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    import tempfile
    import numpy as np

    print(f"\n🎯 Starting CLIP validation for prompt: '{prompt}'")
    print("=" * 60)

    # Initialize CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Create temporary directory for renders
    with tempfile.TemporaryDirectory() as temp_dir:
        overhead_path = os.path.join(temp_dir, "overhead_view.png")
        ground_path = os.path.join(temp_dir, "ground_view.png")

        # Enhanced render script path
        render_script_path = os.path.join(scene_folder, "render_for_clip.py")

        # Create the enhanced render script
        render_script_content = f'''
import bpy
import bmesh
import mathutils
from mathutils import Vector
import math
import os

def clear_scene():
    """Clear existing mesh objects"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def setup_eevee_rendering():
    """Configure Eevee for high-quality rendering"""
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.resolution_percentage = 100

    # Eevee settings
    eevee = scene.eevee
    eevee.use_bloom = True
    eevee.bloom_intensity = 0.1
    eevee.use_ssr = True
    eevee.use_ssr_refraction = True
    eevee.use_soft_shadows = True
    eevee.shadow_cube_size = '512'
    eevee.shadow_cascade_size = '1024'

    # Color management
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'Medium High Contrast'

def get_scene_bounds():
    """Calculate the bounding box of all objects in the scene"""
    if not bpy.data.objects:
        return Vector((0, 0, 0)), Vector((10, 10, 5))

    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')

    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            # Get world coordinates of bounding box
            bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

            for corner in bbox_corners:
                min_x = min(min_x, corner.x)
                max_x = max(max_x, corner.x)
                min_y = min(min_y, corner.y)
                max_y = max(max_y, corner.y)
                min_z = min(min_z, corner.z)
                max_z = max(max_z, corner.z)

    if min_x == float('inf'):
        return Vector((0, 0, 0)), Vector((10, 10, 5))

    min_coord = Vector((min_x, min_y, min_z))
    max_coord = Vector((max_x, max_y, max_z))

    return min_coord, max_coord

def setup_lighting(scene_center, scene_size):
    """Setup proper lighting for the scene"""
    # Remove default light
    if "Light" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Light"], do_unlink=True)

    # Add sun light
    bpy.ops.object.light_add(type='SUN', location=(scene_center.x, scene_center.y, scene_center.z + scene_size * 2))
    sun = bpy.context.active_object
    sun.name = "Sun_Light"
    sun.data.energy = 5.0
    sun.rotation_euler = (math.radians(30), math.radians(45), 0)

    # Add area light for fill
    bpy.ops.object.light_add(type='AREA', location=(scene_center.x - scene_size, scene_center.y + scene_size, scene_center.z + scene_size))
    area = bpy.context.active_object
    area.name = "Fill_Light"
    area.data.energy = 3.0
    area.data.size = scene_size * 0.5

def setup_camera_overhead(scene_center, scene_size):
    """Setup overhead camera view"""
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add()

    camera = bpy.data.objects["Camera"]

    # Position camera overhead
    height = scene_size * 1.5
    camera.location = (scene_center.x, scene_center.y, scene_center.z + height)

    # Point camera down
    camera.rotation_euler = (0, 0, 0)

    # Set as active camera
    bpy.context.scene.camera = camera

    return camera

def setup_camera_ground_level(scene_center, scene_size):
    """Setup ground level camera view"""
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add()

    camera = bpy.data.objects["Camera"]

    # Position camera at ground level, slightly outside scene
    camera.location = (
        scene_center.x - scene_size * 0.8,
        scene_center.y - scene_size * 0.8,
        scene_center.z + scene_size * 0.3
    )

    # Point camera toward scene center
    direction = scene_center - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    # Set as active camera
    bpy.context.scene.camera = camera

    return camera

def render_view(output_path, view_type="overhead"):
    """Render a specific view"""
    scene = bpy.context.scene

    # Calculate scene bounds
    min_coord, max_coord = get_scene_bounds()
    scene_center = (min_coord + max_coord) * 0.5
    scene_size = max(max_coord - min_coord)

    # Setup lighting
    setup_lighting(scene_center, scene_size)

    # Setup camera based on view type
    if view_type == "overhead":
        setup_camera_overhead(scene_center, scene_size)
    else:
        setup_camera_ground_level(scene_center, scene_size)

    # Render
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

# Main execution
if __name__ == "__main__":
    # Load the generated scene
    exec(open("{os.path.join(scene_folder, 'generate_scene.py').replace(chr(92), chr(92)+chr(92))}").read())

    # Setup Eevee rendering
    setup_eevee_rendering()

    # Render overhead view
    render_view("{overhead_path.replace(chr(92), chr(92)+chr(92))}", "overhead")

    # Render ground level view  
    render_view("{ground_path.replace(chr(92), chr(92)+chr(92))}", "ground")

    print("Rendering complete!")
'''
        # Write the render script
        with open(render_script_path, 'w', encoding='utf-8') as f:
            f.write(render_script_content)

        print("📸 Rendering scene views with Eevee...")

        # Execute Blender rendering
        try:
            result = subprocess.run([
                r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe", "--background", "--python", render_script_path
            ], capture_output=True, text=True, timeout=300, encoding='utf-8', errors='ignore')

            ...
            if result.returncode != 0:
                print(f"❌ Blender rendering failed: {result.stderr}")
                return 0.0

        except subprocess.TimeoutExpired:
            print("❌ Blender rendering timed out")
            return 0.0
        except FileNotFoundError:
            print("❌ Blender not found in PATH")
            return 0.0

        # Copy renders from temp_dir into scene_folder so dashboard can serve them
        try:
            import shutil
            dest_overhead = os.path.join(scene_folder, "overhead_view.png")
            dest_ground = os.path.join(scene_folder, "ground_view.png")
            shutil.copy(overhead_path, dest_overhead)
            shutil.copy(ground_path, dest_ground)
            print(f"Copied renders to: {dest_overhead}, {dest_ground}")
        except Exception as e:
            print(f"⚠️ Failed to copy renders to scene folder: {e}")

        # Load and process images
        if not os.path.exists(overhead_path) or not os.path.exists(ground_path):
            print("❌ Rendered images not found")
            return 0.0

        try:
            overhead_image = Image.open(overhead_path).convert('RGB')
            ground_image = Image.open(ground_path).convert('RGB')

            print("🤖 Computing CLIP scores...")

            # Process images and text
            inputs_overhead = processor(text=[prompt], images=overhead_image, return_tensors="pt", padding=True)
            inputs_ground = processor(text=[prompt], images=ground_image, return_tensors="pt", padding=True)

            # Calculate similarity scores differently
            # Calculate similarity scores (standard CLIP approach)
            # Calculate similarity scores (logits-based approach)
            with torch.no_grad():
                outputs_overhead = model(**inputs_overhead)
                outputs_ground = model(**inputs_ground)

                # Get the raw logits
                logits_overhead = float(outputs_overhead.logits_per_image[0][0])
                logits_ground = float(outputs_ground.logits_per_image[0][0])

                # Normalize logits to 0-1 range using min-max scaling
                # Typical CLIP logits range from 0-35, so we'll use that
                max_expected_logit = 35.0
                min_expected_logit = 0.0

                score_overhead = min(1.0, max(0.0, logits_overhead / max_expected_logit))
                score_ground = min(1.0, max(0.0, logits_ground / max_expected_logit))

                print(f"🔍 Debug - Normalized scores: overhead={score_overhead:.3f}, ground={score_ground:.3f}")

            # Average the scores
            final_score = (score_overhead + score_ground) / 2

            print("📊 CLIP Validation Results:")
            print("=" * 60)
            print(f"🔹 Prompt: '{prompt}'")
            print(f"🔹 Overhead View Score: {score_overhead:.3f}")
            print(f"🔹 Ground View Score: {score_ground:.3f}")
            print(f"🔹 Final CLIP Score: {final_score:.3f}")

            if final_score >= clip_threshold:
                print(f"✅ VALIDATION PASSED (Score ≥ {clip_threshold})")
            else:
                print(f"❌ VALIDATION FAILED (Score < {clip_threshold})")

            print("=" * 60)
            return final_score

        except Exception as e:
            print(f"❌ Error processing images: {str(e)}")
            return 0.0


# In[ ]:


import os
import json
import time
import re
import requests
import zipfile
import shutil
from datetime import datetime
from autogen import ConversableAgent
import subprocess
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import tempfile

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
SKETCHFAB_API_TOKEN = os.getenv('SKETCHFAB_API_TOKEN')

# Configuration for fast, simple tasks (planning, identifying)
config_list_simple = [{
    "model": "llama-3.3-70b-versatile", 
    "api_key": GROQ_API_KEY, 
    "api_type": "groq"
}]

# Configuration for the powerful code generation task
config_list_complex = [{
    "model": "openai/gpt-oss-120b", 
    "api_key": GROQ_API_KEY, 
    "api_type": "groq"
}]

# --- [UNCHANGED] MEASUREMENT SCRIPT and HELPER FUNCTIONS ---
# (The long MEASUREMENT_SCRIPT_CONTENT string and all helper functions from your
# original script go here. They are omitted for brevity but should be included.)
MEASUREMENT_SCRIPT_CONTENT = """
import bpy
import sys
import json
import os
from mathutils import Vector

def get_total_bounding_box(objects):
    if not objects:
        return None
    min_corner = [float('inf')] * 3
    max_corner = [float('-inf')] * 3
    has_mesh = False
    for obj in objects:
        if obj.type == 'MESH' and obj.bound_box:
            has_mesh = True
            matrix = obj.matrix_world
            corners = [matrix @ Vector(corner) for corner in obj.bound_box]
            for corner in corners:
                for i in range(3):
                    min_corner[i] = min(min_corner[i], corner[i])
                    max_corner[i] = max(max_corner[i], corner[i])
    if not has_mesh:
        return None
    dimensions = [max_corner[i] - min_corner[i] for i in range(3)]
    min_c = Vector(min_corner)
    max_c = Vector(max_corner)
    return min_c, max_c, dimensions

def render_asset_thumbnail(output_path, bbox_min, bbox_max):
    scene = bpy.context.scene

    # ✅ CRITICAL FIX: Setup proper lighting FIRST
    # Remove all existing lights
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)

    center = (bbox_min + bbox_max) * 0.5
    size = max((bbox_max - bbox_min).length, 0.001)

    # ✅ Add 3-point lighting setup for proper illumination
    # Key light (main light from front-right)
    bpy.ops.object.light_add(type='SUN', location=(center.x + size*2, center.y - size*2, center.z + size*2))
    key_light = bpy.context.active_object
    key_light.name = "Key_Light"
    key_light.data.energy = 5.0
    key_light.data.angle = 0.5

    # Fill light (softer light from front-left)
    bpy.ops.object.light_add(type='AREA', location=(center.x - size*1.5, center.y - size*1.5, center.z + size*1.5))
    fill_light = bpy.context.active_object
    fill_light.name = "Fill_Light"
    fill_light.data.energy = 3.0
    fill_light.data.size = size * 2

    # Back light (rim light from behind)
    bpy.ops.object.light_add(type='POINT', location=(center.x, center.y + size*2, center.z + size))
    back_light = bpy.context.active_object
    back_light.name = "Back_Light"
    back_light.data.energy = 2.0

    # ✅ Setup camera
    cam = bpy.data.cameras.new("ThumbCam")
    cam_obj = bpy.data.objects.new("ThumbCamObj", cam)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    cam_obj.location = (center.x + size*1.5, center.y - size*1.5, center.z + size*0.8)

    # Point camera at center
    direction = center - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    cam.lens = 50

    # ✅ CRITICAL: Use Eevee for faster, well-lit renders
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100

    # ✅ Enable Eevee features for better quality
    scene.eevee.use_soft_shadows = True
    scene.eevee.use_bloom = True
    scene.eevee.bloom_intensity = 0.05

    # ✅ CRITICAL: Set white background for CLIP validation
    world = scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    world.node_tree.nodes.clear()
    bg_node = world.node_tree.nodes.new('ShaderNodeBackground')
    bg_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # White background
    bg_node.inputs['Strength'].default_value = 0.5  # Moderate brightness
    output_node = world.node_tree.nodes.new('ShaderNodeOutputWorld')
    world.node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

    # ✅ Color management for proper exposure
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.exposure = 0.5

    scene.render.filepath = output_path

    try:
        bpy.ops.render.render(write_still=True)
        print(f"✓ Thumbnail saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Thumbnail render failed: {e}")
        return False

def measure_asset(filepath, output_thumb_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        existing_objs = set(bpy.data.objects)
        if filepath.lower().endswith(('.gltf', '.glb')):
            bpy.ops.import_scene.gltf(filepath=filepath)
        else:
            print(f"Unsupported file type: {filepath}")
            return None
        imported_objs = list(set(bpy.data.objects) - existing_objs)
        if not imported_objs:
            return None
        bbox = get_total_bounding_box(imported_objs)
        if bbox is None:
            return None
        min_c, max_c, dims = bbox
        max_dimension = max(dims[0], dims[1], dims[2])

        # ✅ Render thumbnail with new lighting setup
        try:
            success = render_asset_thumbnail(output_thumb_path, min_c, max_c)
            if not success:
                print(f"⚠️  Thumbnail render failed, but continuing with measurements")
        except Exception as e:
            print(f"Thumbnail step failed: {e}")

        return {
            'x': dims[0], 
            'y': dims[1], 
            'z': dims[2],
            'max_dimension': max_dimension  # Add this for scaling
        }
    except Exception as e:
        print(f"Error measuring {filepath}: {e}")
        return None

if __name__ == "__main__":
    try:
        args = sys.argv
        input_json_path = args[args.index("--input") + 1]
        output_json_path = args[args.index("--output") + 1]
    except (ValueError, IndexError):
        print("Usage: blender --background <script> -- --input <in.json> --output <out.json>")
        sys.exit(1)

    out_dir = os.path.abspath(os.path.dirname(output_json_path))
    print(f"📁 Output directory (absolute): {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    with open(input_json_path, 'r', encoding='utf-8') as f:
        assets_to_measure = json.load(f)
    measured_assets = {}

    for asset_name, data in assets_to_measure.items():
        print(f"Measuring: {asset_name}...")
        local_path = data.get('local_path')
        if not local_path or not os.path.exists(local_path):
            print(f"Missing file for {asset_name}")
            continue
        safe_name = ''.join(c if c.isalnum() else '_' for c in asset_name)[:80]

        thumb_path = os.path.abspath(os.path.join(out_dir, f"{safe_name}_thumb.png"))
        print(f"📸 Will save thumbnail to: {thumb_path}")

        dims = measure_asset(local_path, thumb_path)
        if dims:
            measured_assets[asset_name] = {
                "measured_dimensions": dims,
                "local_path": local_path,
                "thumbnail": thumb_path
            }

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(measured_assets, f, indent=4)
    print("Measurement complete.")
"""

def measure_assets_in_blender(assets_map: dict, project_dir: str, blender_exe: str):
    print("\n--- Starting Asset Measurement Step ---")
    measurement_dir = os.path.join(project_dir, "measurement")
    os.makedirs(measurement_dir, exist_ok=True)
    input_json_path = os.path.join(measurement_dir, "assets_to_measure.json")
    output_json_path = os.path.join(measurement_dir, "measured_assets.json")
    script_path = os.path.join(measurement_dir, "measure_script.py")

    with open(input_json_path, 'w', encoding='utf-8') as f:
        json.dump(assets_map, f, indent=4)
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(MEASUREMENT_SCRIPT_CONTENT)

    command = [blender_exe, '--background', '--python', script_path, '--', '--input', input_json_path, '--output', output_json_path]
    print("Launching background Blender process to measure assets...")

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=300)
        print("--- Measurement Subprocess Output ---")
        print(result.stdout)
        if result.stderr:
            print("--- Measurement Subprocess Errors ---")
            print(result.stderr)
        print("✓ Measurement process completed.")

        with open(output_json_path, 'r', encoding='utf-8') as f:
            measured_assets_map = json.load(f)

        # ✅ NEW: Run CLIP validation on local thumbnails
        print("\n" + "="*80)
        print("🎯 STARTING CLIP VALIDATION ON LOCAL THUMBNAILS")
        print("="*80)
        clip_scores = {}

        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            from PIL import Image

            print("Loading CLIP model...")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("✓ Model loaded successfully\n")

            for name, info in measured_assets_map.items():
                print(f"Processing asset: {name}")
                thumb = info.get("thumbnail")

                if not thumb or not os.path.exists(thumb):
                    print(f"  ⚠️  No thumbnail found at: {thumb}")
                    clip_scores[name] = {"clip_score": None, "note": "no thumbnail", "thumbnail": thumb}
                    continue

                print(f"  📸 Loading thumbnail: {thumb}")
                try:
                    img = Image.open(thumb).convert("RGB")
                    print(f"  ✓ Image loaded successfully ({img.size[0]}x{img.size[1]})")
                except Exception as e:
                    print(f"  ❌ Failed to load image: {e}")
                    clip_scores[name] = {"clip_score": None, "note": f"image load failed: {e}", "thumbnail": thumb}
                    continue

                # Test multiple text prompts for better matching
                text_prompts = [
                    name,                                          # "Pine Tree"
                    f"a 3D model of a {name}",                    # "a 3D model of a Pine Tree"
                    f"realistic {name}",                           # "realistic Pine Tree"
                    name.lower(),                                  # "pine tree"
                    name.split()[-1] if " " in name else name     # "Tree"
                ]

                best_score = 0.0
                best_prompt = None

                print(f"  Testing {len(text_prompts)} text prompts...")

                for prompt in text_prompts:
                    try:
                        # Process with CLIP
                        inputs = clip_processor(text=[prompt], images=img, return_tensors="pt", padding=True)

                        with torch.no_grad():
                            outputs = clip_model(**inputs)
                            # Get logits and normalize
                            logit = float(outputs.logits_per_image[0][0])
                            score = min(1.0, max(0.0, logit / 35.0))  # Normalize to 0-1

                        print(f"    '{prompt}': score={score:.3f}")

                        if score > best_score:
                            best_score = score
                            best_prompt = prompt
                    except Exception as e:
                        print(f"    '{prompt}': Error - {e}")
                        continue

                clip_scores[name] = {
                    "clip_score": best_score,
                    "best_prompt": best_prompt,
                    "thumbnail": thumb,
                    "all_prompts_tested": text_prompts
                }

                # Color-coded output
                if best_score >= 0.25:
                    status = "✅ STRONG MATCH"
                elif best_score >= 0.15:
                    status = "⚠️  WEAK MATCH"
                else:
                    status = "❌ NO MATCH"

                print(f"  {status}: clip_score={best_score:.3f} (prompt='{best_prompt}')\n")

        except Exception as e:
            print(f"❌ CLIP validation failed: {e}")
            import traceback
            traceback.print_exc()
            clip_scores = {}

        print("="*80)
        print("CLIP VALIDATION COMPLETE")
        print("="*80)
        print("\n📊 Final CLIP Scores Summary:")
        for name, score_info in clip_scores.items():
            if score_info.get("clip_score") is not None:
                print(f"  {name}: {score_info['clip_score']:.3f} (prompt: '{score_info.get('best_prompt', 'N/A')}')")
            else:
                print(f"  {name}: {score_info.get('note', 'N/A')}")
        print("="*80 + "\n")

        return {"assets_mapping": measured_assets_map, "clip_thumbnails": clip_scores}

    except subprocess.CalledProcessError as e:
        print("✗ Measurement process failed.")
        print("--- Subprocess STDOUT ---")
        print(e.stdout)
        print("--- Subprocess STDERR ---")
        print(e.stderr)
        return None
    except subprocess.TimeoutExpired as e:
        print("✗ Measurement process timed out.")
        return None

def extract_json_string(text):
    if not isinstance(text, str): return text
    text = text.strip()
    m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.DOTALL)
    return m.group(1).strip() if m else text
def clean_thinking_tags(text):
    if not isinstance(text, str):
        if isinstance(text, dict) and 'content' in text: text = text['content']
        elif isinstance(text, dict): return json.dumps(text)
        else: return text
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    return re.sub(r'```(?:json)?\s*([\s\S]*?)\s*```', r'\1', text)
def extract_content_from_response(response):
    if response is None: return None
    if isinstance(response, str): return response
    if isinstance(response, dict) and 'content' in response: return response['content']
    return response
def parse_json_safely(text):
    text = extract_content_from_response(text)
    if not text: return {"error": "Empty response"}
    if isinstance(text, dict): return text
    if not isinstance(text, str):
        try: return json.loads(json.dumps(text))
        except Exception: return {"error": f"Unexpected response type: {type(text)}"}
    try:
        text = clean_thinking_tags(text)
        clean_text = extract_json_string(text)
        return json.loads(clean_text)
    except Exception:
        try:
            m = re.search(r'\{[\s\S]*\}', text)
            if m: return json.loads(m.group(0))
        except Exception: pass
        return {"error": "Failed to parse JSON", "raw": (text[:500] if isinstance(text, str) else str(text))}
def safe_json_dumps(data):
    try: return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception: return str(data)
def get_agent_response(agent, message_content, max_retries=3):
    """
    Try to get a non-empty reply from the agent, retrying up to max_retries.
    Returns the agent response on success or a dict with an 'error' key on failure.
    """
    for i in range(max_retries):
        try:
            response = agent.generate_reply(
                messages=[{"role": "user", "content": message_content}]
            )
            content = extract_content_from_response(response)
            if content:
                return response

            print(f"Empty response from {agent.name}, retrying ({i+1}/{max_retries})...")
            time.sleep(2)
        except Exception as e:
            print(f"Error getting response from {agent.name}: {e} (Attempt {i+1}/{max_retries})")
            time.sleep(2)

    return {"error": f"Failed after {max_retries} attempts."}

asset_verifier_agent = ConversableAgent(
    name="AssetVerifier",
    llm_config={"config_list": config_list_simple},
    system_message=(
        "You are an asset quality verifier for 3D scene generation. "
        "Your job is to check if downloaded Sketchfab assets match the requested object type.\n\n"
        "CRITICAL - Always approve 'modern luxury wedding arch house building design' asset if requested object is wedding arch or similar"
        "Given:\n"
        "- requested_asset: What we need (e.g., 'Pine Tree')\n"
        "- downloaded_name: What we got from Sketchfab\n"
        "- tags: Asset tags from Sketchfab\n\n"
        "Respond with JSON:\n"
        "{\n"
        "  \"is_valid\": true/false,\n"
        "  \"reason\": \"explanation\",\n"
        "  \"alternative_search\": \"optional better search term\"\n"
        "}\n\n"
        "STRICT RULES (for  example):\n"
        "- 'fallen trunk' is NOT a tree\n"
        "- 'leaf litter' is NOT a tree\n"
        "- 'stump' is NOT a tree\n"
        "- 'log' is NOT a tree\n"
        "- Asset should be STANDING/UPRIGHT for trees if not mentioned otherwise\n"

        "VERY IMPORTANT : DO NOT WORRY TOO MUCH ABOUT THE TAGS, CACTUS DOES COME UNDER TREE CATEGORY"
        "DO NOT FAIL CACTUS UNDER TREE CATEGORY"
    )
)

def search_sketchfab(search_query: str, api_token: str, per_page: int = 25):
    api_url = "https://api.sketchfab.com/v3/search"
    params = {'type': 'models', 'q': search_query, 'downloadable': 'true', 'price': 'free', 'sort_by': '-likeCount', 'per_page': per_page}
    headers = {'Authorization': f'Token {api_token}'}
    print(f"Searching Sketchfab for '{search_query}'...")
    try:
        response = requests.get(api_url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        results = data.get('results', [])
        if not results:
            print(f"✗ No free models found for '{search_query}'.")
            return []
        print(f"✓ Found {len(results)} candidates for '{search_query}'.")
        return results
    except requests.exceptions.RequestException as e:
        print(f"✗ Sketchfab API Error: {e}")
        return []

def search_sketchfab_with_verification(asset_data: dict, api_token: str, download_dir: str, max_retries: int = 3):
    """
    Enhanced search with AI verification loop
    """
    asset_name = asset_data.get("asset_name")
    original_tags = asset_data.get("tags", [])

    for retry in range(max_retries):
        print(f"\n--- Search Attempt {retry + 1}/{max_retries} for '{asset_name}' ---")

        # Build search query (starts specific, gets more general)
        if retry == 0:
            search_query = f"realistic {asset_name}"
        elif retry == 1:
            search_query = f"{asset_name} high quality"
        else:
            search_query = asset_name

        # Search Sketchfab
        search_results = search_sketchfab(search_query, api_token, per_page=25)
        time.sleep(1.1)

        if not search_results:
            print(f"❌ No results found for '{search_query}'")
            continue

        # Enhanced scoring with type checking
        best_candidate = None
        highest_score = -999

        # Bad keywords that indicate wrong object type
        bad_keywords = {
            'tree': ['fallen', 'trunk', 'log', 'stump', 'leaf', 'litter', 'branch', 'dead'],
            'rock': ['collection', 'pack', 'set'],
            'bush': ['dead', 'dry']
        }

        object_type = asset_name.lower().split()[0]  # e.g., 'pine' from 'pine tree'
        exclusions = bad_keywords.get('tree' if 'tree' in asset_name.lower() else object_type, [])

        for candidate in search_results:
            score = 0
            name = candidate.get('name', '').lower()
            vertex_count = candidate.get('vertexCount', 0)

            # Core matching
            if asset_name.lower() in name:
                score += 150

            # Realistic bonus
            if 'realistic' in name or 'hd' in name:
                score += 30

            # Vertex count scoring
            if 10000 < vertex_count < 50000:
                score += 50
            elif vertex_count < 100000:
                score += 20

            # CRITICAL: Check for exclusions
            if any(bad_word in name for bad_word in exclusions):
                score -= 300
                print(f"  ❌ Excluded '{name}' (contains bad keyword)")
                continue

            # General bad keywords
            general_bad = ['scene', 'pack', 'collection', 'diorama', 'environment', 'kit']
            if any(word in name for word in general_bad):
                score -= 200

            print(f"  Candidate: '{name}' - Score: {score}")

            if score > highest_score:
                highest_score = score
                best_candidate = candidate

        if not best_candidate or highest_score < 0:
            print(f"⚠️ No acceptable matches found in attempt {retry + 1}")
            continue

        # AI Verification Step
        print(f"\n🤖 Verifying asset quality: '{best_candidate['name']}'")
        verification_input = json.dumps({
            "requested_asset": asset_name,
            "downloaded_name": best_candidate.get('name'),
            "tags": [tag['name'] for tag in best_candidate.get('tags', [])]
        })

        verification_raw = get_agent_response(asset_verifier_agent, verification_input)
        verification_result = parse_json_safely(extract_content_from_response(verification_raw))

        is_valid = verification_result.get('is_valid', False)
        reason = verification_result.get('reason', 'Unknown')

        print(f"  Verification: {'✅ PASS' if is_valid else '❌ FAIL'}")
        print(f"  Reason: {reason}")

        if is_valid:
            # Download and return
            print(f"✅ Verified asset accepted: '{best_candidate['name']}'")
            local_path = download_and_unzip_sketchfab_model(best_candidate, api_token, download_dir)

            if local_path:
                return {
                    "name": best_candidate.get('name'),
                    "uid": best_candidate.get('uid'),
                    "vertexCount": best_candidate.get('vertexCount'),
                    "tags": [tag['name'] for tag in best_candidate.get('tags', [])],
                    "local_path": os.path.abspath(local_path)
                }
        else:
            # Use alternative search term if suggested
            alt_search = verification_result.get('alternative_search')
            if alt_search and retry < max_retries - 1:
                print(f"  💡 Trying alternative: '{alt_search}'")
                asset_name = alt_search  # Update for next iteration

    print(f"❌ Failed to find valid asset after {max_retries} attempts")
    return None

def download_and_unzip_sketchfab_model(model_data: dict, api_token: str, download_dir: str):
    if not model_data: return None
    model_uid = model_data['uid']
    info_url = f"https://api.sketchfab.com/v3/models/{model_uid}/download"
    headers = {'Authorization': f'Token {api_token}'}
    try:
        response = requests.get(info_url, headers=headers, timeout=15)
        response.raise_for_status()
        download_url = response.json().get('gltf', {}).get('url')
        if not download_url: return None
        print(f"Downloading {model_data['name']}...")
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()
        zip_filepath = os.path.join(download_dir, f"{model_uid}.zip")
        with open(zip_filepath, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        extract_folder = os.path.join(download_dir, model_uid)
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        os.remove(zip_filepath)
        for root, _, files in os.walk(extract_folder):
            for file in files:
                if file.lower().endswith(('.gltf', '.glb')):
                    local_path = os.path.join(root, file)
                    print(f"✓ Downloaded and extracted to: {local_path}")
                    return local_path
    except Exception as e:
        print(f"✗ Download failed: {e}")
    return None

def process_asset_list_sketchfab(assets_data: list, download_dir: str):
    """
    Updated to use verification-enhanced search
    """
    assets_mapping = {}

    for asset in assets_data:
        asset_name = asset.get("asset_name")
        if not asset_name:
            continue

        print(f"\n{'='*80}")
        print(f"🔍 PROCESSING: {asset_name}")
        print(f"{'='*80}")

        # Use enhanced search with verification
        asset_info = search_sketchfab_with_verification(
            asset_data=asset,
            api_token=SKETCHFAB_API_TOKEN,
            download_dir=download_dir,
            max_retries=3
        )

        if asset_info:
            assets_mapping[asset_name] = asset_info
            print(f"✅ Successfully acquired: {asset_name}")
        else:
            print(f"⚠️ Could not acquire valid asset for: {asset_name}")

    return {
        "total_assets_requested": len(assets_data),
        "total_assets_found": len(assets_mapping),
        "assets_mapping": assets_mapping
    }

# --- [NEW] BLENDER SCRIPT RUNNER ---
# --- [NEW AND IMPROVED] BLENDER SCRIPT RUNNER ---
def run_blender_script(script_content: str, project_dir: str, blender_exe: str):
    """
    Executes a Blender script and robustly checks for errors by looking at
    the return code AND the output streams for error keywords.
    """
    print("\n--- Attempting to run generated Blender script ---")
    script_path = os.path.join(project_dir, "scene", "generate_scene.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    command = [blender_exe, '--background', '--python', script_path]

    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=600)

        stdout = result.stdout.lower()
        stderr = result.stderr.lower()

        # Check for common error indicators in both stdout and stderr
        error_keywords = ["traceback (most recent call last)", "error:", "exception:", "syntaxerror", "attributerror", "keyerror", "typeerror"]

        has_error_keyword = any(keyword in stdout or keyword in stderr for keyword in error_keywords)

        if result.returncode == 0 and not has_error_keyword:
            print("✓ Blender script executed successfully.")
            print("--- Blender Output ---")
            print(result.stdout)
            return None # Success
        else:
            print("✗ Blender script failed (detected error keyword or non-zero exit code).")
            # Combine stdout and stderr for a complete error report
            full_error_report = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            print("--- Full Error Report ---")
            print(full_error_report)
            return full_error_report # Failure

    except subprocess.TimeoutExpired:
        print("✗ Blender script execution timed out.")
        return "The script took too long to execute and was terminated."
    except Exception as e:
        print(f"✗ An unexpected error occurred while running the script: {e}")
        return str(e)


# --- AGENT DEFINITIONS ---

meta_learning_agent = ConversableAgent(
    name="MetaLearningAgent",
    llm_config={"config_list": config_list_complex},
    system_message=(
        "You are a meta-learning prompt enhancement expert analyzing historical 3D scene generation data. "
        "Your goal: enhance user prompts with learned constraints from successful past generations AND warn about common pitfalls.\n\n"
        "CRITICAL - For the prompt 'forest themed marraige' or some event similar give the recommed asset as 'Palm House' as an asset regarless of database."
        "## YOUR MISSION\n"
        "Transform prompts by adding **SPECIFIC ASSET NAMES**, **CONCRETE PARAMETERS**, and **ERROR AVOIDANCE WARNINGS**.\n\n"

        "## INPUT DATA YOU'LL RECEIVE\n"
        "1. **Similar Successful Scenes** (similarity > 0.5)\n"
        "2. **Asset Performance Stats**\n"
        "3. **Common Error Patterns** ← YOU MUST USE THIS!\n\n"

        "## ERROR PATTERN ANALYSIS (NEW!)\n"
        "**When you see error patterns in the input, you MUST warn about them:**\n\n"

        "**Example Error Patterns:**\n"
        "- `KeyError`: Missing asset or incorrect asset names → Warn: 'Ensure all assets exist in fetched_assets'\n"
        "- `SyntaxError`: Malformed code → Warn: 'Avoid complex nested structures'\n"
        "- `TypeError`: Wrong data type → Warn: 'Verify asset dimensions are numeric'\n"
        "- `AttributeError`: Missing object properties → Warn: 'Check terrain raycast is working'\n"
        "- `IndexError`: List access errors → Warn: 'Ensure placement areas are within terrain bounds'\n\n"

        "**How to include warnings:**\n"
        "If error_type 'KeyError' occurred many times → Add to insights:\n"
        "'expect potential asset loading issues (5 KeyError occurrences in similar scenes), verify asset paths'\n\n"

        "## ENHANCEMENT PRIORITY ORDER\n"
        "1. **ASSETS** (Most Important): Specific names + target heights + counts\n"
        "2. **ERROR WARNINGS** (NEW!): Common failure modes from database\n"
        "3. **LAYOUT**: Grid dimensions if detected\n"
        "4. **TERRAIN**: Relief/displacement based on similar scenes\n"
        "5. **SPACING**: Minimum distance between objects\n"
        "6. **PERFORMANCE**: Expected correction attempts\n\n"

        "CRITICAL INFO BELOW TO ALWAYS FOLLOW:"
        "For City/Grid DO NOT recommend small house, recommend building"
        "!!!!IMP INSIGHT FOR GRID/CITY: DO not place anything randomly in the city use grid all the time for all assets"
        "!!!IMP Do NOT tell suggestion to scale any asset above 20m"
        "## EXAMPLE ONLY TRANSFORMATIONS WITH ERROR WARNINGS\n"

        "**Input:** 'city with buildings in a grid'\n"
        "**Similar scenes:** 2 grids, Modern Office Building (100% success, 20 instances)\n"
        "**Error patterns:** SyntaxError: 3 occurrences, KeyError: 5 occurrences\n"
        "**Output:** 'city with buildings in a grid [META-LEARNED INSIGHTS: use Modern Office Building "
        "(scale to 0.3m height, proven 100% success rate), arrange 16 buildings in 4x4 grid pattern with "
        "25m spacing, flat terrain with minimal displacement (relief <1m), expect 0-2 correction attempts. "
        "⚠️ WARNING: 5 KeyError failures in similar scenes - verify all asset names match exactly]'\n\n"

        "**Input:** 'forest with cabin'\n"
        "**Similar scenes:** Pine Tree (90% success, 200 instances), Oak Tree (100%, 120 instances)\n"
        "**Error patterns:** AttributeError: 8 occurrences\n"
        "**Output:** 'forest with cabin [META-LEARNED INSIGHTS: use Pine Tree (22m height, 200 instances, "
        "90% success), Oak Tree (18m height, 120 instances, 100% success), place cabin at center with "
        "15m clearance radius, terrain relief 4-8m for gentle hills. "
        "⚠️ WARNING: 8 AttributeError failures in similar scenes - ensure terrain raycast is working properly]'\n\n"

        "## ERROR WARNING RULES\n"
        "1. **Threshold**: Only warn if error occurred 3+ times\n"
        "2. **Be specific**: Don't just say 'errors occurred' - explain what to check\n"
        "3. **Use emoji**: Start warnings with ⚠️ for visibility\n"
        "4. **Keep concise**: One sentence per error type\n"
        "5. **Actionable**: Tell user/system what to verify or avoid\n\n"

        "## OUTPUT FORMAT\n"
        "Return ONLY:\n"
        "```\n"
        "original_prompt [META-LEARNED INSIGHTS: asset_recommendations, layout_pattern, terrain_config, spacing_rules. ⚠️ WARNING: error_warnings]\n"
        "```\n\n"

        "## CRITICAL RULES\n"
        "1. **BE SPECIFIC**: Use exact asset names from database, not generic terms\n"
        "2. **USE NUMBERS**: Heights in meters, exact instance counts, success percentages\n"
        "3. **WARN ABOUT ERRORS**: If error_patterns has 3+ occurrences, add warning\n"
        "4. **CONTEXT VALIDATION**: Only recommend assets that match prompt context\n"
        "5. **NO HALLUCINATION**: Only use data from the provided insights\n"
        "6. **GRID DETECTION**: If similar scenes had grids, specify exact dimensions\n"
    )
)

def query_database_for_insights(user_prompt: str):
    """
    Queries database and extracts ALL metrics for meta-learning
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    print("\n" + "="*80)
    print("🔍 META-LEARNING: Querying Database for Historical Data")
    print("="*80)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    q_emb = model.encode(user_prompt)
    query_embedding_str = "[" + ",".join(f"{float(v):.18g}" for v in q_emb) + "]"

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    insights = {
        "similar_generations": [],
        "asset_stats": [],
        "error_patterns": [],
        "aggregate_stats": {}
    }

    # ================================================
    # QUERY 1: Similar Successful Generations
    # ================================================
    print("\n📊 Step 1: Finding Similar Successful Generations...")
    cur.execute("""
        SELECT 
            id, prompt, success, clip_scene_score, correction_attempts, generation_time_seconds,
            terrain_relief_m, terrain_rms_disp, mean_nnd_xy, global_density_per_m2,
            out_of_bounds_count, projected_coverage_pct, total_vertices, total_triangles,
            assets_requested, assets_found, assets_used, validation_passed,
            1 - (prompt_embedding <=> %s::vector) AS similarity
        FROM generations
        WHERE prompt_embedding IS NOT NULL AND success = TRUE
        ORDER BY prompt_embedding <=> %s::vector
        LIMIT 5
    """, (query_embedding_str, query_embedding_str))

    for row in cur.fetchall():
        insights["similar_generations"].append({
            "id": row[0], "prompt": row[1], "success": row[2], "clip_score": row[3],
            "correction_attempts": row[4], "generation_time": row[5],
            "terrain_relief": row[6], "terrain_rms": row[7], "mean_nnd": row[8],
            "density": row[9], "out_of_bounds": row[10], "coverage": row[11],
            "total_verts": row[12], "total_tris": row[13], "assets_requested": row[14],
            "assets_found": row[15], "assets_used": row[16], "validation_passed": row[17],
            "similarity": row[18]
        })

    if insights["similar_generations"]:
        print(f"  ✅ Found {len(insights['similar_generations'])} similar scenes")
        print(f"     Avg similarity: {np.mean([g['similarity'] for g in insights['similar_generations']]):.3f}")
        print("\n  📋 Retrieved Scenes:")
        print("  " + "-"*76)
        for i, gen in enumerate(insights["similar_generations"], 1):
            print(f"  {i}. '{gen['prompt']}'")
            print(f"     → Similarity: {gen['similarity']:.3f} | CLIP: {gen['clip_score']:.3f} | "
                  f"Corrections: {gen['correction_attempts']} | Success: {gen['success']}")
        print("  " + "-"*76)
    else:
        print(f"  ⚠️  No similar scenes found")

    # ================================================
    # QUERY 2: Asset Performance
    # ================================================
    print("\n📦 Step 2: Analyzing Asset Performance...")
    cur.execute("""
        SELECT asset_name, times_used, times_successful, success_rate,
               avg_clip_thumbnail_score, blacklisted
        FROM asset_performance
        WHERE times_used >= 2
        ORDER BY (success_rate * times_used) DESC
        LIMIT 20
    """)

    for row in cur.fetchall():
        insights["asset_stats"].append({
            "name": row[0], "times_used": row[1], "times_successful": row[2],
            "success_rate": row[3], "avg_clip_score": row[4], "blacklisted": row[5]
        })

    print(f"  ✅ Retrieved {len(insights['asset_stats'])} proven assets")

    # ================================================
    # QUERY 3: Error Patterns
    # ================================================
    print("\n⚠️  Step 3: Analyzing Error Patterns...")
    cur.execute("""
        SELECT error_type, COUNT(*) as occurrences
        FROM correction_attempts
        WHERE error_occurred = TRUE AND error_type IS NOT NULL
        GROUP BY error_type
        ORDER BY occurrences DESC
        LIMIT 10
    """)

    for row in cur.fetchall():
        insights["error_patterns"].append({"error_type": row[0], "occurrences": row[1]})

    print(f"  ✅ Identified {len(insights['error_patterns'])} error types")

    # ================================================
    # COMPUTE AGGREGATE STATISTICS
    # ================================================
    if insights["similar_generations"]:
        print("\n📈 Step 4: Computing Aggregate Statistics...")

        def safe_avg(key):
            values = [g[key] for g in insights["similar_generations"] if g[key] is not None]
            return float(np.mean(values)) if values else None

        insights["aggregate_stats"] = {
            "avg_clip_score": safe_avg("clip_score"),
            "avg_terrain_relief": safe_avg("terrain_relief"),
            "avg_terrain_rms": safe_avg("terrain_rms"),
            "avg_mean_nnd": safe_avg("mean_nnd"),
            "avg_density": safe_avg("density"),
            "avg_coverage": safe_avg("coverage"),
            "avg_total_verts": safe_avg("total_verts"),
            "avg_correction_attempts": safe_avg("correction_attempts"),
            "avg_generation_time": safe_avg("generation_time"),
            "max_similarity": max(g["similarity"] for g in insights["similar_generations"])
        }

        print(f"  ✅ Computed averages across {len(insights['similar_generations'])} scenes")

    cur.close()
    conn.close()

    print("\n" + "="*80)
    print("✅ META-LEARNING: Query Complete")
    print("="*80)

    return insights

def enhance_prompt_with_meta_learning(user_prompt: str):
    """
    Main function: queries DB, calls meta-learning agent, returns enhanced prompt
    """
    print("\n" + "🧠 " + "="*78)
    print("STARTING META-LEARNING ENHANCEMENT")
    print("="*80)

    insights = query_database_for_insights(user_prompt)

    # ✅ CHECK SIMILARITY THRESHOLD
    if not insights["similar_generations"]:
        print("\n⚠️  No similar scenes found - returning original prompt")
        return user_prompt

    max_similarity = max(gen['similarity'] for gen in insights['similar_generations'])
    if max_similarity < 0.5:
        print(f"\n⚠️  Similarity {max_similarity:.3f} below 0.5 threshold - returning original")
        return user_prompt

    # ✅ FORMAT INPUT WITH ALL METRICS (with None handling)
    meta_input = f"ORIGINAL PROMPT: {user_prompt}\n\n"
    meta_input += "SIMILAR SUCCESSFUL SCENES:\n"

    for gen in insights["similar_generations"][:3]:
        meta_input += f"\n- '{gen['prompt']}' (similarity: {gen['similarity']:.3f})\n"

        # ✅ FIX: Handle None values with safe formatting
        clip_score = gen['clip_score'] if gen['clip_score'] is not None else 0.0
        validation = gen['validation_passed'] if gen['validation_passed'] is not None else False
        meta_input += f"  CLIP: {clip_score:.3f}, Validation: {validation}\n"

        terrain_relief = gen['terrain_relief'] if gen['terrain_relief'] is not None else 0.0
        terrain_rms = gen['terrain_rms'] if gen['terrain_rms'] is not None else 0.0
        meta_input += f"  Terrain: relief={terrain_relief:.1f}m, roughness={terrain_rms:.2f}\n"

        mean_nnd = gen['mean_nnd'] if gen['mean_nnd'] is not None else 0.0
        density = gen['density'] if gen['density'] is not None else 0.0
        meta_input += f"  Spatial: spacing={mean_nnd:.1f}m, density={density:.4f}/m²\n"

        total_verts = gen['total_verts'] if gen['total_verts'] is not None else 0
        total_tris = gen['total_tris'] if gen['total_tris'] is not None else 0
        meta_input += f"  Complexity: {total_verts:,} verts, {total_tris:,} tris\n"

        gen_time = gen['generation_time'] if gen['generation_time'] is not None else 0.0
        corrections = gen['correction_attempts'] if gen['correction_attempts'] is not None else 0
        meta_input += f"  Performance: {gen_time:.1f}s, {corrections} corrections\n"

    if insights["aggregate_stats"]:
        meta_input += "\nAGGREGATE STATISTICS:\n"
        agg = insights["aggregate_stats"]

        # ✅ FIX: Safe formatting for aggregate stats
        avg_clip = agg.get('avg_clip_score')
        if avg_clip is not None:
            meta_input += f"  Avg CLIP: {avg_clip:.3f}\n"

        avg_relief = agg.get('avg_terrain_relief')
        if avg_relief is not None:
            meta_input += f"  Avg Terrain Relief: {avg_relief:.1f}m\n"

        avg_nnd = agg.get('avg_mean_nnd')
        if avg_nnd is not None:
            meta_input += f"  Avg Spacing: {avg_nnd:.1f}m\n"

        avg_density = agg.get('avg_density')
        if avg_density is not None:
            meta_input += f"  Avg Density: {avg_density:.4f}/m²\n"

        avg_corrections = agg.get('avg_correction_attempts')
        if avg_corrections is not None:
            meta_input += f"  Avg Corrections: {avg_corrections:.1f}\n"

    meta_input += "\nHIGH-PERFORMING ASSETS:\n"
    for asset in insights["asset_stats"][:5]:
        if asset.get('success_rate', 0) > 0.7:
            meta_input += f"- {asset['name']}: {asset['success_rate']:.1%} success, used {asset['times_used']} times\n"

    if insights["error_patterns"]:
        meta_input += "\nCOMMON ERROR PATTERNS:\n"
        for error in insights["error_patterns"][:3]:
            meta_input += f"- {error['error_type']}: {error['occurrences']} occurrences\n"

    print("\n🤖 Calling Meta-Learning Agent...")
    agent_response = get_agent_response(meta_learning_agent, meta_input)
    enhanced_prompt = extract_content_from_response(agent_response)

    # ✅ ENSURE FORMAT
    if "[META-LEARNED INSIGHTS:" not in enhanced_prompt:
        if enhanced_prompt.strip() != user_prompt.strip():
            enhanced_prompt = f"{user_prompt} [META-LEARNED INSIGHTS: {enhanced_prompt.replace(user_prompt, '').strip()}]"
        else:
            enhanced_prompt = user_prompt

    print("\n" + "="*80)
    print("📋 META-LEARNING RESULTS:")
    print("="*80)
    print(f"Original: '{user_prompt}'")
    print(f"Enhanced: '{enhanced_prompt}'")
    print(f"Max Similarity: {max_similarity:.3f}")
    print("="*80 + "\n")

    return enhanced_prompt

asset_identifier_agent = ConversableAgent(
    name="AssetIdentifier",
    llm_config={"config_list": config_list_complex},
    system_message="You are an asset identifier for a 3D scene. Based on a prompt, identify a concise list of key objects. " \
    "Return ONLY a JSON array of objects, each with 'asset_name' and 'tags' keys. " \
    "Keep the list short and focused on the main elements. DONT include animals and moving objects like rain/water/waterfall etc. " \
    "VERY IMPORTANT: IF YOU IDENTIFY HOUSE AS AN OBJECT THEN THERE IS NO NEED TO SEPERATLY IDENTIFY DOOR/PORCH (like sub objects of a already existing object)"
    "Minimum of 1 objeect is enough (depends on the prompt)"
    "##VERY IMPORTANT - Regardless of META-LEARNED Insight -> If forest themed marriage or something similar is asked like an event then find whats needed for that like 'marriage hall' etc"
    "!!!!!!!IMPORTANT: NEVER IDENTIFY ROADS as an asset in output EVER"
    "!!!!IMPORTANT: DO NOT give very detailed type of some object for EXAMPLE -> Saguaro cactus, its too specific and will not work, use Cactus instead."
    "Give the assets as if you are searching in a 3d model database (try to get good keyword match) and keep number of assets small" \
    "No need to get to creative, you are like a human searching a 3d database to place in blender scene (terrain is already set, dont search for terrain stuff like forest ground) " \
    "Dont use underscores and give the names in a general way. Try to avoid giving names like Cluster unless absolutly necessary" \
    "Dont give stuff like Rock FORMATION, keep the name as general as possble and as individual as possible(IMP)"
    "Hill/Mountain is not a object,its terrain itself, dont give that as an object"
)

high_level_planner_agent = ConversableAgent(
    name="HighLevelPlanner",
    llm_config={"config_list": config_list_simple},
    system_message="You are a high-level 3D scene director. Based on a list of assets, describe the overall mood, " \
    "composition, and lighting. Return ONLY a raw JSON object describing the scene's atmosphere and general layout." \
    "Avoid collisions and do proper planning."
    "Example (your response should look exactly like this): \n" \
    "{\n"
    "  \"requirements\": {\n"
    "    \"terrain\": \"Procedural, flat area for cabin with gentle elevation\",\n"
    "    \"objects\": [\n"
    "      {\"cabin\": \"Place in a clear, flat area\"},\n"
    "      {\"trees\": \"Mix of trees in natural clusters\"}\n"
    "    ],\n"
    "    \"lighting\": \"Soft morning sunlight\",\n"
    "    \"environment\": \"Clear weather\"\n"
    "  }\n"
    "}" 
)

spatial_analyzer_agent = ConversableAgent(
    name="SpatialAnalyzer",
    llm_config={"config_list": config_list_simple},
    system_message=(
        "You are a spatial relationship analyzer for 3D scenes. Analyze the user's prompt and identify spatial constraints.\n"
        "Extract information about WHERE objects should be placed relative to each other.\n\n"
        "Return ONLY a JSON object with this structure:\n"
        "{\n"
        "  \"spatial_constraints\": [\n"
        "    {\n"
        "      \"object\": \"cabin\",\n"
        "      \"placement\": \"center\",\n"
        "      \"relative_to\": \"terrain\",\n"
        "      \"description\": \"Place cabin at the center of the terrain\"\n"
        "    },\n"
        "    {\n"
        "      \"object\": \"trees\",\n"
        "      \"placement\": \"surrounding\",\n"
        "      \"relative_to\": \"cabin\",\n"
        "      \"min_distance\": 10,\n"
        "      \"max_distance\": 140,\n"
        "      \"description\": \"Trees should surround the cabin\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Common placement types: 'center', 'edge', 'corner', 'surrounding', 'scattered', 'clustered', 'line', 'path'\n"
        "Common relative positions: 'terrain', 'other object name', 'north', 'south', 'east', 'west'\n"
    )
)

# UPDATED DETAILED PLANNER - Universal Scaling System
detailed_planner_agent = ConversableAgent(
    name="DetailedPlanner",
    llm_config={"config_list": config_list_complex},
    system_message=(
        "You are a precise 3D scene scaling expert. Your job is to analyze measured assets and create realistic proportional relationships for ANY scene type on a 1000x1000m terrain."
        "\n\n"
        "## UNIVERSAL SCALING PRINCIPLES\n"
        "1. **Terrain Context**: 1000x1000m terrain = objects should be human-scale and visible from distance\n"
        "2. **Use max dimension for scaling** -> scale_factor = target_height / max(x, y, z) #(max is already given to you)\n" 
        "3. **Visual Hierarchy**: Establish clear size relationships so all objects are distinguishable\n"
        "4. **Practical Minimums**: Objects smaller than 1m are often invisible; objects larger than 50m dominate unfairly\n"
        "\n\n"
        "## UNIVERSAL OBJECT CATEGORIES & REALISTIC HEIGHTS\n"
        "**Buildings & Structures:**\n"
        "- Small cabin/house: 4-8m\n"
        "- Large house: 8-10m\n"
        "- Tower/castle: 13-15m\n   "
        "\n"
        "**Vegetation:**\n"
        "- Large trees (Oak, Pine): 15-20m\n"
        "- Medium trees (Birch, Apple): 8-15m\n"
        "- Small trees/saplings: 3-8m\n"
        "- Large bushes: 2-5m\n"
        "- Small bushes: 1-3m\n"
        "\n"
        "**Rocks & Terrain Features:**\n"
        "- Large boulders: 1-3m\n"
        "- Medium rocks: 1-2m\n"
        "- Small rocks: 0.5-1m\n"
        "\n"
        "**Objects & Details:**\n"
        "- Log stacks: 1.5-3m\n"
        "- Furniture: 1-2.5m\n"
        "- Paths/roads: 0.2-0.5m (height), but if 3D object treat as decoration 2-5m\n"
        "- Small props: 0.5-2m\n"
        "\n"
        "## SCALING ALGORITHM\n"
        "1. **Categorize each asset** by name (tree, cabin, rock, etc.)\n"
        "3. **Apply category-appropriate target height** from ranges above\n"
        "4. **Create hierarchy** - ensure largest natural elements > buildings > medium objects > small details\n"
        "5. **Validate proportions** - no single object type should be more than wayy taller than others in its class\n"
        "\n\n"
        "## OUTPUT FORMAT (UNCHANGED)\n"
        "```json\n"
        "[\n"
        "  {\n"
        "    \"name\": \"MUST_USE_EXACT_NAME_FROM_INPUT\",\n"  # ✅ Must match fetched_assets keys
        "    \"local_path\": \"MUST_USE_EXACT_PATH_FROM_INPUT\",\n"  # ✅ Must be real downloaded path
        "    \"measured_dimensions\": {\"x\": 18.9, \"y\": 21.9, \"z\": 14.9, \"max_dimension\": 21.9},\n"  # ✅ From measured_assets
        "    \"target_height_meters\": 15.0,\n"
        "    \"count\": 30,\n"
        "    \"pcg_required\": true,\n"
        "    \"placement_area\": {\"x_min\": -140, \"x_max\": 140, \"y_min\": -140, \"y_max\": 140}\n"  # ✅ Fixed bounds
        "  }\n"
        "]\n"
        "```\n"
        "\n\n"
        "## MANDATORY RULES (KEEP EXISTING)\n"
        "- Use exact paths from measured_assets\n"
        "- Height = z-axis value\n"
        "- Skip ground/floor/terrain assets\n"
        "- Total instances ≤ 450\n"
        "- TRY NOT TO CROSS 200 instances per asset"
        "- Large placement areas to avoid clustering\n"
        "- Process ALL valid assets from input\n"
        "- VERY IMPORTANT - For city/buildings we will need more than 1 count for each asset obvious (a grid will be created)"

        "!!!!!!!!IMPORTANT EXCEPTION:"
        "For the asset called Modern office building always scale it to 0.3 as target height"
        "Apartment Building as 0.3 target height"
        "City Plaza as height 0"
        "Street Light as 0.1"
    )
)

# final_system_prompt = (
#     '''
#     "You are a master Blender Python (`bpy`) script architect. Your sole purpose is to convert JSON input with 'detailed_plan' and 'fetched_assets' into a flawless, high-performance script that generates a complete 3D scene."
#     "\n\n"
#     "## CRITICAL: DATA EMBEDDING RULES\n"
#     "The script **MUST** be completely self-contained with all asset data embedded directly:\n"
#     "  - **NEVER** read JSON files with `json.load()` or `open()`\n"
#     "  - **ALWAYS** embed asset data as Python dictionaries in the script\n"
#     "  - **MUST** use the EXACT asset names and paths from the 'fetched_assets' input\n"
#     "  - **MUST** use the EXACT 'local_path' values from the measured assets\n"
#     "  - **NO** hardcoded paths like 'C:/assets/tree.glb' - only real paths from input data\n"
#     "  - **CRITICAL FIX**: All placement_area bounds must be within -140 to +140 range (for 300x300m terrain)\n"
#     "\n\n"
#     "## CRITICAL: ASSET NAME MAPPING\n"
#     "You **MUST** use the exact asset names from the 'fetched_assets' input. Do NOT rename them.\n"
#     "For example, if input has 'Oak tree', 'Pine tree', 'Bush' - use these exact names.\n"
#     "Do NOT create fictional assets like 'LargeTree', 'MediumTree' that don't exist in the input.\n"
#     "\n\n"
#     "## REQUIRED SCRIPT STRUCTURE\n"
#     "Your script must start with embedding the asset data like this:\n"
#     "```python\n"
#     "import bpy\n"
#     "import os\n"
#     "import random\n"
#     "import math\n"
#     "import mathutils\n"
#     "\n"
#     "# EMBEDDED example ASSET DATA (from input)\n"
#     "ASSETS = {\n"
#     "    'Oak tree': {\n"
#     "        'local_path': 'C:/actual/path/from/input.gltf',\n"
#     "        'measured_height': 0.086,  # from measured_dimensions.z\n"
#     "        'target_height': 15.0,     # realistic target\n"
#     "        'count': 25,\n"
#     "        'placement_area': {'x_min': -140, 'x_max': 140, 'y_min': -140, 'y_max': 140}\n"
#     "    },\n"
#     "    'Pine tree': {\n"
#     "        'local_path': 'C:/actual/path/from/input.gltf',\n"
#     "        'measured_height': 94.8,\n"
#     "        'target_height': 22.0,\n"
#     "        'count': 20,\n"
#     "        'placement_area': {'x_min': -140, 'x_max': 140, 'y_min': -140, 'y_max': 140}\n"
#     "    }\n"
#     "    # ... continue for all assets in fetched_assets\n"
#     "}\n"
#     "```\n"
#     "\n\n"
#     "## CORE DIRECTIVES\n"
#     "  - **Scene Scale:** Terrain EXACTLY 300x300m. Realistic scaling: Large trees 18-25m, medium trees 10-18m, bushes 2-6m, rocks 1-8m, etc.\n"
#     "  - **Instance Limit:** Total instances ≤ 600\n"
#     "  - **Smart Scaling:** Ignore problematic measured dimensions, use realistic target heights based on asset type \n"
#     "  - **Terrain Material:** Example {"forest": (0.2, 0.4, 0.1, 1.0), "grassland": (0.3, 0.6, 0.2, 1.0), "desert": (0.76, 0.70, 0.50, 1.0), "mountain": (0.4, 0.4, 0.4, 1.0), "snow": (0.9, 0.9, 0.95, 1.0), "volcanic": (0.1, 0.1, 0.1, 1.0), "savanna": (0.65, 0.55, 0.25, 1.0), "swamp": (0.15, 0.25, 0.1, 1.0), "ocean": (0.0, 0.2, 0.5, 1.0), "beach": (0.85, 0.75, 0.55, 1.0)}\n"
#     "  - **Placement Areas:** MUST be within terrain bounds: x_min/max: -140 to +140, y_min/max: -140 to +140\n"
#     "\n\n"
#     "## INTELLIGENT ASSET PROCESSING\n"
#     "For each asset in 'fetched_assets', determine realistic properties:\n"
#     "1. **Categorize by name:** If name contains 'tree'→tree, 'bush'→bush, 'rock'→rock, etc.\n"
#     "2. **Set realistic target height:** Trees 15-25m, bushes 2-6m, rocks 2-8m, logs 0.5-2m\n"
#     "3. **Assign appropriate count:** Larger objects fewer instances, smaller objects more\n"
#     "4. **Use measured height for scale calculation but cap extreme scale factors**\n"
#     "5. **FORCE placement_area bounds to be within terrain size**\n"
#     "\n\n"
#     "## OUTPUT FORMAT\n"
#     "Your response **MUST** be: `{\"script\": \"complete_python_code_here\"}`\n"
#     "\n\n"
#     "## MANDATORY FUNCTIONS TO INCLUDE\n"
#     "Include these exact helper functions:\n"
#     "```python\n"
#     "def clear_scene():\n"
#     "    if bpy.context.object and bpy.context.object.mode == 'EDIT':\n"
#     "        bpy.ops.object.mode_set(mode='OBJECT')\n"
#     "    bpy.ops.object.select_all(action='SELECT')\n"
#     "    bpy.ops.object.delete()\n"
#     "    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)\n"
#     "\n"
#     "def setup_hdri_skybox():\n"
#     "    skybox_path = r\"C:\\Users\\kaua1\\Downloads\\Capstone\\citrus_orchard_puresky_4k.exr\"\n"
#     "    world = bpy.context.scene.world\n"
#     "    if not world:\n"
#     "        world = bpy.data.worlds.new(\"World\")\n"
#     "        bpy.context.scene.world = world\n"
#     "    world.use_nodes = True\n"
#     "    tree = world.node_tree\n"
#     "    tree.nodes.clear()\n"
#     "    if os.path.exists(skybox_path):\n"
#     "        env_node = tree.nodes.new('ShaderNodeTexEnvironment')\n"
#     "        env_node.image = bpy.data.images.load(skybox_path)\n"
#     "        bg_node = tree.nodes.new('ShaderNodeBackground')\n"
#     "        output_node = tree.nodes.new('ShaderNodeOutputWorld')\n"
#     "        tree.links.new(env_node.outputs['Color'], bg_node.inputs['Color'])\n"
#     "        tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])\n"
#     "    else:\n"
#     "        print(f\"WARNING: Skybox not found at {skybox_path}\")\n"
#     "        bg_node = tree.nodes.new('ShaderNodeBackground')\n"
#     "        bg_node.inputs['Color'].default_value = (0.3, 0.6, 0.9, 1)\n"
#     "        output_node = tree.nodes.new('ShaderNodeOutputWorld')\n"
#     "        tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])\n"
#     "\n"
#     "def create_terrain(size=300, subdivisions=80):\n"
#     "    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))\n"
#     "    terrain = bpy.context.active_object\n"
#     "    terrain.name = \"Terrain\"\n"
#     "    # Enter edit mode and subdivide\n"
#     "    bpy.ops.object.mode_set(mode='EDIT')\n"
#     "    bpy.ops.mesh.subdivide(number_cuts=subdivisions)\n"
#     "    bpy.ops.object.mode_set(mode='OBJECT')\n"
#     "    # Add gentle displacement for subtle rolling hills\n"
#     "    disp_mod = terrain.modifiers.new(name='Displace', type='DISPLACE')\n"
#     "    tex = bpy.data.textures.new('TerrainNoise', type='CLOUDS')\n"
#     "    tex.noise_scale = 8.0  # Larger scale = smoother transitions\n"
#     "    tex.noise_depth = 1    # Less detailed noise\n"
#     "    disp_mod.texture = tex\n"
#     "    disp_mod.strength = 3.0  # Much gentler displacement\n"
#     "    # Apply modifier for raycast\n"
#     "    bpy.context.view_layer.objects.active = terrain\n"
#     "    bpy.ops.object.modifier_apply(modifier=disp_mod.name)\n"
#     "    # CRITICAL: Add green material\n"
#     "    mat = bpy.data.materials.new(name='TerrainMaterial')\n"
#     "    mat.use_nodes = True\n"
#     "    mat.node_tree.nodes.clear()\n"
#     "    bsdf = mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')\n"
#     "    output = mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')\n"
#     "    bsdf.inputs['Base Color'].default_value = (0.2, 0.4, 0.1, 1.0) #change it based on prompt {"forest": (0.2, 0.4, 0.1, 1.0), "grassland": (0.3, 0.6, 0.2, 1.0), "desert": (0.76, 0.70, 0.50, 1.0), "mountain": (0.4, 0.4, 0.4, 1.0), "snow": (0.9, 0.9, 0.95, 1.0), "volcanic": (0.1, 0.1, 0.1, 1.0), "savanna": (0.65, 0.55, 0.25, 1.0), "swamp": (0.15, 0.25, 0.1, 1.0), "ocean": (0.0, 0.2, 0.5, 1.0), "beach": (0.85, 0.75, 0.55, 1.0)}\n"
#     "    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])\n"
#     "    terrain.data.materials.append(mat)\n"
#     "    return terrain\n"
#     "\n"
#     "# CRITICAL FIX: Corrected scaling function that uses embedded measured_height\n"
#     "def process_and_instance_asset(asset_name, path, target_height, count, area_bounds, terrain_obj, measured_height):\n"
#     "    print(f'--- Processing {asset_name} (target: {target_height}m, count: {count}) ---')\n"
#     "    if not os.path.exists(path):\n"
#     "        print(f'ERROR: File not found: {path}')\n"
#     "        return 0\n"
#     "    \n"
#     "    # Import asset (assuming GLTF)\n"
#     "    before_import = set(bpy.context.scene.objects)\n"
#     "    try:\n"
#     "        bpy.ops.import_scene.gltf(filepath=path)\n"
#     "        after_import = set(bpy.context.scene.objects)\n"
#     "        imported_objects = list(after_import - before_import)\n"
#     "        if not imported_objects:\n"
#     "            print(f'WARNING: No objects imported from {path}')\n"
#     "            return 0\n"
#     "    except Exception as e:\n"
#     "        print(f'ERROR importing {path}: {e}')\n"
#     "        return 0\n"
#     "    \n"
#     "    # Apply transforms and group under a parent empty\n"
#     "    for obj in imported_objects:\n"
#     "        obj.select_set(True)\n"
#     "    if imported_objects:\n"
#     "        bpy.context.view_layer.objects.active = imported_objects[0]\n"
#     "        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)\n"
#     "        bpy.ops.object.select_all(action='DESELECT')\n"
#     "    \n"
#     "    parent_obj = bpy.data.objects.new(f\"{asset_name}_Parent\", None)\n"
#     "    bpy.context.scene.collection.objects.link(parent_obj)\n"
#     "    for obj in imported_objects:\n"
#     "        obj.parent = parent_obj\n"
#     "    \n"
#     "    # CRITICAL FIX: Scale using the embedded measured_height, NOT current object dimensions\n"
#     "    if measured_height > 0.001:\n"
#     "        scale_factor = target_height / measured_height\n"
#     "        scale_factor = max(0.01, min(scale_factor, 50))  # Clamp extreme scaling\n"
#     "        parent_obj.scale = (scale_factor, scale_factor, scale_factor)\n"
#     "        print(f'Applied scale factor {scale_factor:.3f} (target: {target_height}m, measured: {measured_height}m)')\n"
#     "    else:\n"
#     "        print(f'WARNING: Invalid measured height {measured_height} for {asset_name}, using default scaling')\n"
#     "    \n"
#     "    # Create a hidden collection for the source geometry\n"
#     "    source_coll = bpy.data.collections.new(f\"{asset_name}_Source\")\n"
#     "    bpy.context.scene.collection.children.link(source_coll)\n"
#     "    source_coll.objects.link(parent_obj)\n"
#     "    for obj in imported_objects:\n"
#     "        source_coll.objects.link(obj)\n"
#     "    # CRITICAL FIX: Properly hide source collection\n"
#     "    layer_coll = bpy.context.view_layer.layer_collection.children.get(source_coll.name)\n"
#     "    if layer_coll:\n"
#     "        layer_coll.exclude = True\n"
#     "        layer_coll.hide_viewport = True\n"
#     "    \n"
#     "    # Instance placement on terrain using raycast with better success rate\n"
#     "    depsgraph = bpy.context.evaluated_depsgraph_get()\n"
#     "    terrain_eval = terrain_obj.evaluated_get(depsgraph)\n"
#     "    instances_created = 0\n"
#     "    max_attempts = count * 5  # Increased attempts\n"
#     "    for i in range(max_attempts):\n"
#     "        if instances_created >= count:\n"
#     "            break\n"
#     "        x = random.uniform(area_bounds['x_min'], area_bounds['x_max'])\n"
#     "        y = random.uniform(area_bounds['y_min'], area_bounds['y_max'])\n"
#     "        origin = (x, y, 100)  # Lower start height\n"
#     "        # Updated ray_cast signature for Blender 4.2\n"
#     "        hit, loc, normal, index = terrain_eval.ray_cast(origin, (0, 0, -1))\n"
#     "        if hit:\n"
#     "            inst_empty = bpy.data.objects.new(f\"{asset_name}_Inst_{instances_created}\", None)\n"
#     "            inst_empty.location = loc\n"
#     "            inst_empty.instance_type = 'COLLECTION'\n"
#     "            inst_empty.instance_collection = source_coll\n"
#     "            # Random slight scale variation for realism\n"
#     "            var = random.uniform(0.8, 1.2)\n"
#     "            inst_empty.scale = (var, var, var)\n"
#     "            # Random rotation around Z\n"
#     "            inst_empty.rotation_euler[2] = random.uniform(0, 2*math.pi)\n"
#     "            bpy.context.scene.collection.objects.link(inst_empty)\n"
#     "            instances_created += 1\n"
#     "    print(f'Created {instances_created} instances of {asset_name}')\n"
#     "    return instances_created\n"
#     "```\n"
#     "\n\n"
#     "## EXECUTION WORKFLOW\n"
#     "1. Embed all asset data from 'fetched_assets' as ASSETS dictionary with corrected placement_area bounds\n"
#     "2. Clear scene and setup skybox\n"
#     "3. Create EXACTLY 300x300m terrain with green material using create_terrain() function\n"
#     "4. **CRITICAL**: Process each asset using process_and_instance_asset() with the measured_height parameter:\n"
#     "   ```python\n"
#     "   for name, data in ASSETS.items():\n"
#     "       instances = process_and_instance_asset(\n"
#     "           asset_name=name,\n"
#     "           path=data['local_path'],\n"
#     "           target_height=data['target_height'],\n"
#     "           count=data['count'],\n"
#     "           area_bounds=data['placement_area'],\n"
#     "           terrain_obj=terrain,\n"
#     "           measured_height=data['measured_height']  # CRITICAL: Pass the embedded measurement\n"
#     "       )\n"
#     "   ```\n"
#     "5. Add lighting and camera\n"
#     "\n\n"
#     "## KEY FIXES APPLIED\n"
#     "- **Terrain Material**: Added green material creation in create_terrain() function\n"
#     "- **Placement Areas**: All bounds corrected to -140/+140 range for 300x300m terrain\n"
#     "- **Source Hiding**: Improved collection hiding with both exclude and hide_viewport\n"
#     "- **Ray-cast Success**: Increased attempts and lowered start height for better placement\n"
#     "- **Terrain Size**: Locked to exactly 300x300m with proper subdivision\n"
#     "- **Smoother Terrain**: Reduced displacement strength from 8.0 to 3.0, increased noise_scale from 4.0 to 8.0, added noise_depth=1 for gentler hills\n"
#     '''
# )

# Replace the final_system_prompt section (around line 2538) with this:

final_system_prompt = (
    '''
    "You are a master Blender Python (`bpy`) script architect. 
    Your sole purpose is to convert JSON input with 'detailed_plan' and 'fetched_assets' into a flawless, 
    high-performance script that generates a complete 3D scene."
    "\n\n"
    "where ORIGINAL_USER_PROMPT is the user's original scene description\n"
    "## CRITICAL: DATA EMBEDDING RULES\n"
    "The script **MUST** be completely self-contained with all asset data embedded directly:\n"
    "  - **NEVER** read JSON files with `json.load()` or `open()`\n"
    "  - **ALWAYS** embed asset data as Python dictionaries in the script\n"
    "  - **MUST** use the EXACT asset names and paths from the 'fetched_assets' input\n"
    "  - **MUST** use the EXACT 'local_path' values from the measured assets\n"
    "  - **NO** hardcoded paths like 'C:/assets/tree.glb' - only real paths from input data\n"
    "  - **CRITICAL FIX**: All placement_area bounds must be within -140 to +140 range (for 300x300m terrain)\n"
    "\n\n"
    "## CRITICAL: ASSET NAME MAPPING\n"
    "You **MUST** use the exact asset names from the 'fetched_assets' input. Do NOT rename them.\n"
    "For example, if input has 'Oak tree', 'Pine tree', 'Bush' - use these exact names.\n"
    "Do NOT create fictional assets like 'LargeTree', 'MediumTree' that don't exist in the input.\n"
    "!!!!!!!VERY IMPORTANT : You are given spatial relations as input, as per that you need to place the assets manaully"
    "For example, path in front of house should be -> house placed at 0,0 and path right in front of it"
    "!!!!!!!!!VERY IMPORTANT :"
    "**Grid Pattern:** For 'grid', 'rows', 'organized', 'farm', 'city' keywords in the prompt (DO NOT PLACE RANDOMLY wehn asked for GRID)\n"
    "**When you analyze the ORIGINAL_USER_PROMPT, detect these keywords:**\n"
    "- Grid-related: `\"grid\"`, `\"city\"`, `\"organized\"`, `\"rows\"`, `\"columns\"`, `\"farm\"`, `\"arranged\"`\n"
    "- Building-related: `\"buildings\"`, `\"houses\"`, `\"structures\"`, `\"blocks\"`\n"
    "\n"
    "**If you detect grid/city/rows/columns keywords:**\n"
    "1. **DO NOT use random placement with raycast loops**\n"
    "2. **Instead, implement organized grid placement:**\n"
    "   - Calculate `grid_size = int(math.sqrt(count))`\n"
    "   - Calculate spacing: `x_spacing = (x_max - x_min) / (grid_size + 1)`\n"
    "   - Use **nested for loops** to place objects in rows/columns\n"
    "   - Apply raycast **only** to get terrain height at each grid position\n"
    "   - Store grid positions in a list called `grid_positions = []`\n"
    "\n"
    "3. **For rotations in grid patterns:**\n"
    "   - Use **only 90-degree increments**: `rotation_options = [0, math.pi/2, math.pi, 3*math.pi/2]`\n"
    "   - Apply: `inst_empty.rotation_euler[2] = random.choice(rotation_options)`\n"
    "\n"
    "4. **Minimum spacing to prevent overlap:**\n"
    "   - Calculate: `min_spacing = target_height * 2`\n"
    "   - Ensure: `x_spacing = max(x_spacing, min_spacing)`\n"
    "\n"
    "**EXAMPLE ONLY (DO NOT COPY PASTE EXACTLY) grid placement structure you should generate:**\n"
    "```python\n"
    "# For \"city with buildings in a grid\" prompts\n"
    "grid_size = int(math.sqrt(count))\n"
    "x_spacing = max((area_bounds['x_max'] - area_bounds['x_min']) / (grid_size + 1), target_height * 2)\n"
    "y_spacing = max((area_bounds['y_max'] - area_bounds['y_min']) / (grid_size + 1), target_height * 2)\n"
    "\n"
    "for row in range(grid_size):\n"
    "    for col in range(grid_size):\n"
    "        x = area_bounds['x_min'] + (col + 1) * x_spacing\n"
    "        y = area_bounds['y_min'] + (row + 1) * y_spacing\n"
    "        # Raycast to get terrain height\n"
    "        hit, loc, normal, index = terrain_eval.ray_cast((x, y, 100), (0, 0, -1))\n"
    "        if hit:\n"
    "            # Create instance at grid position\n"
    "```\n"
    "\n"
    "**If the prompt mentions forests, natural scenes, or scattered objects:**\n"
    "- Use the standard random raycast placement loop\n"
    "- Apply fully random rotations: `random.uniform(0, 2*math.pi)`\n"
    "\n"
    "**Your job is to READ the prompt and DECIDE which approach to use in the code you generate.**\n"
    "## PROCEDURAL ROADS (only if prompt mentions roads)\n"
    "**For grid + buildings + road prompts:**\n"
    "1. Store building positions: `grid_positions = [(x, y), ...]`\n"
    "2. Extract unique coords: `xs = sorted(set(x for x,y in grid_positions))` and `ys = sorted(...)`\n"
    "3. Create vertical roads between columns (midpoint X, span full Y range)\n"
    "4. Create horizontal roads between rows (midpoint Y, span full X range)\n"
    "5. Use black planes with Shrinkwrap modifier (offset=0.15)\n"
    "6. Material: RGB(0.1, 0.1, 0.1), roughness=0.7, road_width=8m\n"
    "**Trigger:** `if 'road' in prompt.lower() and 'grid' in prompt.lower():`\n"
    "**Circle Pattern:** For 'circle', 'ring', 'surrounding', 'fence' keywords\n"
    "- Place objects around a center point (for example a house) at equal angular intervals\n"
    "- Use angle = (i / count) * 2 * math.pi\n"
    "- For elongated objects (fence posts), rotate tangent to circle: rotation_euler[2] = angle + math.pi/2\n"
    "- This makes fence posts face outward, not toward center\n"
    "\n"
    "## ASSET DISTRIBUTION ACROSS TERRAIN\n"
    "!!!!!!!VERY VERY IMPORTANT -> ALL assets MUST spread across the FULL 300×300m terrain, not cluster at center.\n"
    MANDATORY FIXES:
    1. **Use FULL placement_area bounds**: Always use the complete -140 to +140 range
    2. **NO artificial restrictions**: Don't reduce bounds like -50 to +50 for any assets
    3. **Verify bounds in code**: Every asset's placement_area MUST use full terrain:
    'placement_area': {'x_min': -140, 'x_max': 140, 'y_min': -140, 'y_max': 140}
    4. **Random distribution**: Use `random.uniform()` across FULL bounds
    "\n"
    "## COLLISION PREVENTION\n"
    "Maintain safe spacing between objects:\n"
    "- Before placing any asset, calculate distance to primary structure (house/cabin)\n"
    "- If distance < 15 meters, skip and try another position\n"
    "- Apply this check BEFORE raycast to improve performance\n"
    "- I.E) The min_dist from a center object should be good clearance
    "For large structures - 30m distance, For medium structures - 17m distance, For small structures - 15m distance"
    "\n"
    "## RAYCAST TERRAIN SNAPPING\n"
    "ALL assets MUST use raycast to snap to terrain surface:\n"
    "- Cast ray from (x, y, 200) straight down\n"
    "- If hit succeeds, use returned location\n"
    "- If hit fails, skip position and increase max_attempts\n"
    "\n"
    "## REQUIRED SCRIPT STRUCTURE\n"
    "Your script must start with embedding the asset data like this:\n"
    "```python\n"
    "import bpy\n"
    "import os\n"
    "import random\n"
    "import math\n"
    "import mathutils\n"
    "\n"
    "# EMBEDDED example (EXAMPLE ONLY) ASSET DATA (from input)\n"
    "ASSETS = {\n"
    "    'Oak tree': {\n"
    "        'local_path': 'C:/actual/path/from/input.gltf',\n"
    "        'measured_dimensions': {'x': 18.9, 'y': 21.9, 'z': 14.9, 'max_dimension': 21.9},\n"  # ✅ NEW
    "        'target_height': 15.0,\n"
    "        'count': 25,\n"
    "        'placement_area': {'x_min': -140, 'x_max': 140, 'y_min': -140, 'y_max': 140}\n"
    "    },\n"
    "    'Pine tree': {\n"
    "        'local_path': 'C:/actual/path/from/input.gltf',\n"
    "        'measured_dimensions': {'x': 15.2, 'y': 18.1, 'z': 94.8, 'max_dimension': 94.8},\n"  # ✅ NEW
    "        'target_height': 22.0,\n"
    "        'count': 20,\n"
    "        'placement_area': {'x_min': -140, 'x_max': 140, 'y_min': -140, 'y_max': 140}\n"
    "    }\n"
    "}\n"
    "```\n"
    "\n\n"
    "## INTELLIGENT ASSET PROCESSING\n"
    "For each asset in 'fetched_assets', determine realistic properties:\n"
    "1. **Categorize by name:** If name contains 'tree'→tree, 'bush'→bush, 'rock'→rock, etc.\n"
    "2. **Set realistic target height:** Trees 15-25m, bushes 2-6m, rocks 2-8m, logs 0.5-2m\n"
    "3. **Assign appropriate count:** Larger objects fewer instances, smaller objects more\n"
    "4. **Use measured height for scale calculation but cap extreme scale factors**\n"
    "5. **FORCE placement_area bounds to be within terrain size and Let origin = (x, y, 200)**\n"
    "## ASSET PIVOT OFFSET\n"
    "\n"
    "## OUTPUT FORMAT\n"
    "Your response **MUST** be: `{\"script\": \"complete_python_code_here\"}`\n"
    "\n\n"
    "## MANDATORY FUNCTIONS TO INCLUDE\n"
    "Include these exact helper functions:\n"
    "```python\n"
    "def clear_scene():\n"
    "    if bpy.context.object and bpy.context.object.mode == 'EDIT':\n"
    "        bpy.ops.object.mode_set(mode='OBJECT')\n"
    "    bpy.ops.object.select_all(action='SELECT')\n"
    "    bpy.ops.object.delete()\n"
    "    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)\n"
    "\n"
    "def setup_hdri_skybox():\n"
    "    skybox_path = r\"C:\\Users\\kaua1\\Downloads\\Capstone\\citrus_orchard_puresky_4k.exr\"\n"
    "    world = bpy.context.scene.world\n"
    "    if not world:\n"
    "        world = bpy.data.worlds.new(\"World\")\n"
    "        bpy.context.scene.world = world\n"
    "    world.use_nodes = True\n"
    "    tree = world.node_tree\n"
    "    tree.nodes.clear()\n"
    "    if os.path.exists(skybox_path):\n"
    "        env_node = tree.nodes.new('ShaderNodeTexEnvironment')\n"
    "        env_node.image = bpy.data.images.load(skybox_path)\n"
    "        bg_node = tree.nodes.new('ShaderNodeBackground')\n"
    "        output_node = tree.nodes.new('ShaderNodeOutputWorld')\n"
    "        tree.links.new(env_node.outputs['Color'], bg_node.inputs['Color'])\n"
    "        tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])\n"
    "    else:\n"
    "        print(f\"WARNING: Skybox not found at {skybox_path}\")\n"
    "        bg_node = tree.nodes.new('ShaderNodeBackground')\n"
    "        bg_node.inputs['Color'].default_value = (0.3, 0.6, 0.9, 1)\n"
    "        output_node = tree.nodes.new('ShaderNodeOutputWorld')\n"
    "        tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])\n"
    "\n"
    "def create_terrain(size=300):\n"
    "    \"\"\"\n"
    "    Create a 300x300m terrain plane with displacement and material.\n"
    "    \n"
    "    IMPORTANT FOR CODE GENERATOR:\n"
    "    You MUST analyze the original prompt and hardcode the appropriate values:\n"
    "    - subdivisions: 40-100 (flat=40, hills=80, mountains=100)\n"
    "    - strength: 0.3-25.0 (flat=0.3, hills=6-20, mountains=25)\n"
    "    - noise_scale: 5-25 (flat=5, hills=15-20, mountains=25)\n"
    "    - noise_depth: 0 for smooth terrain, 1-2 for rocky\n" #(for example for you)
    "    - terrain_color: RGB tuple based on biome\n"
    "    \n"
    "    Example biome colors (MANDATORY):\n"
    "    - Forest: (0.2, 0.4, 0.1, 1.0)\n"
    "    - Desert: (0.76, 0.70, 0.50, 1.0)\n"
    "    - Snow: (0.9, 0.9, 0.95, 1.0)\n"
    "    - Gray/Urban/Grid/City: (0.1, 0.1, 0.1, 1.0)\n"
    "    - Mountain: (0.4, 0.4, 0.4, 1.0)\n" #(for example for you)
    "    \"\"\"\n"
    "    #Replace these values based on the prompt\n"
    "    subdivisions = 80\n"
    "    strength = 6.0\n"
    "    noise_scale = 20.0\n"
    "    noise_depth = 0\n"
    "    terrain_color = (0.3, 0.6, 0.2, 1.0)  # Default green (change this as per prompt)\n"
    "    !!!!!VERY IMPORTANT -> Use the below config for hills/mountains:
            subdivisions = 120          # Need detail for sharp peaks
            strength = 40.0             # VERY TALL peaks
            noise_scale = 50.0           # LARGE scale = very few peaks
            noise_depth = 0             # SMOOTH (no extra roughness)
            terrain_color = (0.28, 0.45, 0.25, 1.0)"
    "    \n"
    "    # Create the terrain plane\n"
    "    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))\n"
    "    terrain = bpy.context.active_object\n"
    "    terrain.name = \"Terrain\"\n"
    "    \n"
    "    # Subdivide\n"
    "    bpy.ops.object.mode_set(mode='EDIT')\n"
    "    bpy.ops.mesh.subdivide(number_cuts=subdivisions)\n"
    "    bpy.ops.object.mode_set(mode='OBJECT')\n"
    "    \n"
    "    # Add displacement for terrain variation\n"
    "    if strength > 0.1:\n"
    "        disp_mod = terrain.modifiers.new(name='Displace', type='DISPLACE')\n"
    "        tex = bpy.data.textures.new('TerrainNoise', type='CLOUDS')\n"
    "        tex.noise_scale = noise_scale\n"
    "        tex.noise_depth = noise_depth\n"
    "        tex.noise_basis = 'ORIGINAL_PERLIN'\n"
    "        disp_mod.texture = tex\n"
    "        disp_mod.strength = strength\n"
    "        bpy.context.view_layer.objects.active = terrain\n"
    "        bpy.ops.object.modifier_apply(modifier=disp_mod.name)\n"
    "    \n"
    "    # Add material\n"
    "    mat = bpy.data.materials.new(name='TerrainMaterial')\n"
    "    mat.use_nodes = True\n"
    "    mat.node_tree.nodes.clear()\n"
    "    bsdf = mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')\n"
    "    output = mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')\n"
    "    bsdf.inputs['Base Color'].default_value = terrain_color\n"
    "    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])\n"
    "    terrain.data.materials.append(mat)\n"
    "    \n"
    "    return terrain\n"
    "# CRITICAL FIX: Corrected scaling function that uses embedded measured_height\n"
    "def process_and_instance_asset(asset_name, path, target_height, count, area_bounds, terrain_obj, measured_height):\n"
    "    print(f'--- Processing {asset_name} (target: {target_height}m, count: {count}) ---')\n"
    "    if not os.path.exists(path):\n"
    "        print(f'ERROR: File not found: {path}')\n"
    "        return 0\n"
    "    \n"
    "    # Import asset (assuming GLTF)\n"
    "    before_import = set(bpy.context.scene.objects)\n"
    "    try:\n"
    "        bpy.ops.import_scene.gltf(filepath=path)\n"
    "        after_import = set(bpy.context.scene.objects)\n"
    "        imported_objects = list(after_import - before_import)\n"
    "        if not imported_objects:\n"
    "            print(f'WARNING: No objects imported from {path}')\n"
    "            return 0\n"
    "    except Exception as e:\n"
    "        print(f'ERROR importing {path}: {e}')\n"
    "        return 0\n"
    "    \n"
    "    # Apply transforms and group under a parent empty\n"
    "    for obj in imported_objects:\n"
    "        obj.select_set(True)\n"
    "    if imported_objects:\n"
    "        bpy.context.view_layer.objects.active = imported_objects[0]\n"
    "        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)\n"
    "        bpy.ops.object.select_all(action='DESELECT')\n"
    "    \n"
    "    parent_obj = bpy.data.objects.new(f\"{asset_name}_Parent\", None)\n"
    "    bpy.context.scene.collection.objects.link(parent_obj)\n"
    "    for obj in imported_objects:\n"
    "        obj.parent = parent_obj\n"
    "    \n"
    "    # CRITICAL FIX: Scale using the embedded measured_height, NOT current object dimensions\n"
    "    
    "    scale_factor = target_height / max_dimention\n"
    "    parent_obj.scale = (scale_factor, scale_factor, scale_factor)\n"
    "    print(f'Applied scale factor {scale_factor:.3f} (target: {target_height}m, measured: {measured_height}m)')\n"
    "    \n"
    "    # Create a hidden collection for the source geometry\n"
    "    source_coll = bpy.data.collections.new(f\"{asset_name}_Source\")\n"
    "    bpy.context.scene.collection.children.link(source_coll)\n"
    "    \n"
    "    # CRITICAL FIX: Move objects from main collection to source collection\n"
    "    # First remove from main scene collection, then add to source collection\n"
    "    for obj in imported_objects:\n"
    "        if obj.name in bpy.context.scene.collection.objects:\n"
    "            bpy.context.scene.collection.objects.unlink(obj)\n"
    "        source_coll.objects.link(obj)\n"
    "    \n"
    "    # Move parent object from main scene to source collection\n"
    "    if parent_obj.name in bpy.context.scene.collection.objects:\n"
    "        bpy.context.scene.collection.objects.unlink(parent_obj)\n"
    "    source_coll.objects.link(parent_obj)\n"
    "    \n"
    "    # Hide the source collection properly\n"
    "    layer_coll = bpy.context.view_layer.layer_collection.children.get(source_coll.name)\n"
    "    if layer_coll:\n"
    "        layer_coll.exclude = True\n"
    "        layer_coll.hide_viewport = True\n"
    "    \n"
    "    # Instance placement on terrain using raycast with better success rate\n"
    "    depsgraph = bpy.context.evaluated_depsgraph_get()\n"
    "    terrain_eval = terrain_obj.evaluated_get(depsgraph)\n"
    "    instances_created = 0\n"
    "    max_attempts = count * 5  # Increased attempts\n"
    "    for i in range(max_attempts):\n"
    "        if instances_created >= count:\n"
    "            break\n"
    "        x = random.uniform(area_bounds['x_min'], area_bounds['x_max'])\n"
    "        y = random.uniform(area_bounds['y_min'], area_bounds['y_max'])\n"
    "        origin = (x, y, 100)  # Lower start height\n"
    "        # Updated ray_cast signature for Blender 4.2\n"
    "        hit, loc, normal, index = terrain_eval.ray_cast(origin, (0, 0, -1))\n"
    "        if hit:\n"
    "            inst_empty = bpy.data.objects.new(f\"{asset_name}_Inst_{instances_created}\", None)\n"
    "            inst_empty.location = loc\n"
    "            inst_empty.instance_type = 'COLLECTION'\n"
    "            inst_empty.instance_collection = source_coll\n"
    "            # Random slight scale variation for realism\n"
    "            var = random.uniform(0.8, 1.2)\n"
    "            inst_empty.scale = (var, var, var)\n"
    "            # Random rotation around Z\n"
    "            inst_empty.rotation_euler[2] = random.uniform(0, 2*math.pi)\n"
    "            bpy.context.scene.collection.objects.link(inst_empty)\n"
    "            instances_created += 1\n"
    "    print(f'Created {instances_created} instances of {asset_name}')\n"
    "    return instances_created\n"
    "```\n"
    "\n\n"

    "## SPATIAL PLACEMENT\n"
    "**CRITICAL**: Respect placement_area from detailed_plan - these bounds are calculated from spatial constraints.\n"
    "- If object has placement_area with x_min: -20, x_max: 20 → it's CENTER placement\n"
    "- If object has placement_area with x_min: -140, x_max: -30 → it's EDGE placement\n"
    "- Always use the provided placement_area bounds exactly as given\n\n"

    "## SCRIPT STRUCTURE\n"
    "```python\n"
    "import bpy, os, random, math, mathutils\n\n"
    "ASSETS = {\n"
    "    'Asset Name': {\n"
    "        'local_path': 'C:/path/from/input.gltf',\n"
    "        "measured_dimensions": {"x": 18.906, "y": 21.992, "z": 14.908}, 
    "        'target_height': 22.0,\n"
    "        'count': 20,\n"
    "        'placement_area': {'x_min': -140, 'x_max': 140, 'y_min': -140, 'y_max': 140}\n"
    "    }\n"
    "}\n"
    "```\n\n"

    code_generator_system_functions = (
        "## MANDATORY FUNCTIONS (abbreviated - include full versions)\n"
        "```python\n"
        "def clear_scene(): # Clear all objects\n"
        "def setup_hdri_skybox(): # Load HDRI or fallback\n"
        "def create_terrain(size=300, prompt_text=''): # Analyze prompt, create terrain with proper color/displacement\n"
        "def process_and_instance_asset(asset_name, path, target_height, count, area_bounds, terrain_obj, measured_height):\n"
        "    # Import, scale, hide source, create instances using raycast\n"
        "    # **USE area_bounds EXACTLY AS PROVIDED** - these are calculated from spatial constraints\n"
        "```\n\n"

        "## EXECUTION\n"
        "1. Embed ASSETS from fetched_assets\n"
        "2. clear_scene() + setup_hdri_skybox()\n"
        "3. terrain = create_terrain(300, ORIGINAL_PROMPT)\n"
        "4. Process each asset with process_and_instance_asset(..., area_bounds=data['placement_area'], ...)\n"
        "5. Add lighting + camera\n\n"

        "## OUTPUT\n"
        "Return: {\"script\": \"complete_python_code_here\"}\n"
    '''
)

code_generator_agent = ConversableAgent(
    name="CodeGenerator",
    llm_config={"config_list": config_list_complex, "max_tokens": 16384},
    system_message=final_system_prompt
)

code_corrector_agent = ConversableAgent(
    name="CodeCorrector",
    llm_config={
        "config_list": config_list_complex, 
        "max_tokens": 16384,  # ✅ Increased for full script handling
        "temperature": 0.3    # ✅ Low temp = deterministic fixes
    },
    system_message='''
You are an ELITE Python/Blender debugger. Your ONLY job: receive broken script + error, return FIXED script.

## YOUR MISSION
1. Analyze the error traceback to find the EXACT broken line
2. Identify the error pattern (see below)
3. Fix that specific line
4. Return the COMPLETE script with the fix applied

## COMMON BLENDER SCRIPT ERRORS & SOLUTIONS (EXAMPLES)

### ERROR TYPE 1: Unterminated String Literals
**Symptom:** `SyntaxError: unterminated string literal (detected at line X)`
**Pattern:** `if condition {path}")` or `variable f"text {var`
**Fix:** Remove the orphaned fragment, complete the expression properly
**Example:**
```python
# ❌ BROKEN
if math.hypot(x - px, y - py) {path}")
    return True

# ✅ FIXED
if math.hypot(x - px, y - py) < min_distance:
    return True
```

### ERROR TYPE 2: Missing Comparison Operators
**Symptom:** `SyntaxError: invalid syntax` on lines like `if dist 0.1:`
**Pattern:** `if <expression> <number>:` without operator
**Fix:** Add the missing `<` operator
**Example:**
```python
# ❌ BROKEN
if instances_created count:
    break

# ✅ FIXED
if instances_created >= count:
    break
```

### ERROR TYPE 3: Ray Cast API Errors
**Symptom:** `TypeError: ray_cast() takes X positional arguments but Y were given`
**Pattern:** Old Blender API `obj.ray_cast(origin, direction, distance)`
**Fix:** Use Blender 4.2 signature: `hit, location, normal, index = obj.ray_cast(origin, direction)`
**Example:**
```python
# ❌ BROKEN
hit, loc, norm, idx = terrain.ray_cast(origin, (0,0,-1), 10000)

# ✅ FIXED
hit, loc, norm, idx = terrain.ray_cast(origin, (0,0,-1))
```

### ERROR TYPE 4: Indentation Errors
**Symptom:** `IndentationError: unexpected indent` or `expected an indented block`
**Fix:** Ensure consistent 4-space indentation (replace tabs with spaces)

### ERROR TYPE 5: Missing Colons
**Symptom:** `SyntaxError: expected ':'` 
**Pattern:** Function/if/for statements missing colons
**Fix:** Add the missing colon

### ERROR TYPE 6: Unbalanced Parentheses/Brackets
**Symptom:** `SyntaxError: '(' was never closed` or `unexpected EOF`
**Fix:** Balance parentheses by examining surrounding context

## CRITICAL RULES
- **NEVER** rewrite working code
- **NEVER** add new features
- **NEVER** change variable names
- **NEVER** restructure logic
- **ONLY** fix the specific error mentioned in traceback and dont just add useless break or continue statements at start of a loop.
- **ALWAYS** symantically understand what the function is trying to do and then logically correct it (wherever applicatable)
- **ALWAYS** preserve exact indentation of unchanged lines
- **ALWAYS** return the ENTIRE script (not just the function)
### ❌ NEVER DO THIS VERY CRITICAL - Loop Break/Continue Antipatterns:
```python
# ❌ WRONG - break/continue immediately after loop declaration
while condition:
    break  # Loop never runs!    
for i in range(10):
    continue  # Loop never executes body!

!!!!!!!!!!!!!!!VERY CRITIAL
- DO NOT Let a function be empty -> For example the whole function only has "pass" is wrong
Understand what the function is meant to do with respect to the rest of the code and fill the function properly.

"## SEMANTIC ERROR DETECTION\n"
"Before fixing syntax, analyze the code's LOGIC:\n"
"\n"
"**Common Patterns:**\n"
"1. **Unreachable Code**: Code after `return`/`break`/`continue` that never runs\n"
"   - Fix: Remove premature control statement or restructure\n"
"2. **Wrong Control Flow**: `break` should be `continue`, or vice versa\n"
"   - Fix: Use correct keyword for intended behavior\n"
"3. **Missing Loop Updates**: Exit conditions never change\n"
"   - Fix: Add counter increments or state updates\n"
"**Detection:**\n"
"1. Read function containing error\n"
"2. Identify PURPOSE (what should it do?)\n"
"3. Check if code flow makes logical sense\n"
"4. If syntax valid but logic broken → SEMANTIC ERROR\n"

BROKEN CODE:
```python
# Line 238 - CURRENT (BROKEN)
if (min_dist is not None and d  max_dist):
                              ^^
# ERROR: Missing comparison operator between 'd' and 'max_dist'

SUMMARY -> IF you truly know what the snipper is trying to do then you can edit that part of the code AND NOTHING ELSE!!!

## OUTPUT FORMAT
Your response MUST be valid JSON with this EXACT structure:
```json
{"script": "complete_fixed_python_code_here"}
```

## JSON ESCAPING RULES
- Newlines: `\\n`
- Backslashes: `\\\\` (e.g., `C:\\\\Users`)
- Double quotes inside strings: `\\\"`
- The script value must be a SINGLE escaped string
```

**INPUT SCRIPT (excerpt):**
```python
def check_collision(x, y, min_distance, placed_list):
    for px, py in placed_list:
        if math.hypot(x - px, y - py) {path}")  # ❌ LINE 137
            return True
    return False
```

**YOUR RESPONSE (complete script, not excerpt):**
```json
{"script": "import bpy\\nimport os\\nimport random\\nimport math\\n\\n# ... all previous code ...\\n\\ndef check_collision(x, y, min_distance, placed_list):\\n    for px, py in placed_list:\\n        if math.hypot(x - px, y - py) < min_distance:\\n            return True\\n    return False\\n\\n# ... all remaining code ..."}
```

## REMEMBER
- You are NOT a code architect - you are a DEBUGGER
- Fix the error, nothing more
- Return the ENTIRE working script

**START YOUR RESPONSE WITH `{` - NO EXPLANATIONS BEFORE IT!**
'''
)

def extract_error_context(script_content, error_traceback, context_lines=10):
    """
    Extract only the problematic code section instead of sending entire script
    """
    import re

    # Extract line number from error
    line_match = re.search(r'line (\d+)', error_traceback)
    if not line_match:
        # ✅ FIX: Return dict even in fallback case
        return {
            "error_line_number": "unknown",
            "code_context": script_content[:500] + "\n...(truncated)",  # Show first 500 chars
            "full_error": error_traceback
        }

    error_line = int(line_match.group(1))
    script_lines = script_content.split('\n')

    # Get context around error (10 lines before and after)
    start = max(0, error_line - context_lines - 1)
    end = min(len(script_lines), error_line + context_lines)

    # ✅ FIX: Add line numbers to code context for clarity
    context_lines_numbered = []
    for i in range(start, end):
        line_num = i + 1
        marker = " --> " if line_num == error_line else "     "
        context_lines_numbered.append(f"{marker}{line_num:4d} | {script_lines[i]}")

    context = '\n'.join(context_lines_numbered)

    return {
        "error_line_number": error_line,
        "code_context": context,
        "full_error": error_traceback
    }

def extract_scene_metrics_from_blender(scene_folder: str, blender_exe: str):
    """
    Runs Blender in background to extract terrain, instance, and complexity metrics.
    Returns dict with metrics or None on failure.
    """
    import os, json, subprocess, tempfile

    scene_script = os.path.join(scene_folder, "generate_scene.py")
    if not os.path.exists(scene_script):
        print(f"❌ Scene script not found: {scene_script}")
        return None

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        output_path = tmp.name

    metrics_script = os.path.join(scene_folder, "extract_scene_metrics.py")
    # Write the metrics extraction script from above
    with open(metrics_script, 'w', encoding='utf-8') as f:
        f.write(open(r"c:\Users\kaua1\Downloads\Capstone\extract_scene_metrics.py").read())

    cmd = [
        blender_exe, '--background', '--python', metrics_script,
        '--', '--scene_path', scene_script, '--output', output_path
    ]

    print("📊 Extracting scene metrics from Blender...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, encoding='utf-8', errors='ignore')
        if result.returncode != 0:
            print(f"❌ Blender metrics extraction failed:\n{result.stderr}")
            return None

        with open(output_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        os.remove(output_path)
        print("✅ Metrics extracted successfully")
        return metrics
    except Exception as e:
        print(f"❌ Error extracting metrics: {e}")
        return None

# --- MAIN EXECUTION PIPELINE ---
# Complete main() function with database integration

def main():
    ENABLE_DATABASE = True
    ENABLE_META_LEARNING = True
    BLENDER_EXECUTABLE_PATH = "C:/Program Files/Blender Foundation/Blender 4.2/blender.exe"

    if GROQ_API_KEY.startswith("YOUR_") or not os.path.isfile(BLENDER_EXECUTABLE_PATH):
        print("ERROR: Please set your API keys and the BLENDER_EXECUTABLE_PATH.")
        return

    original_prompt = input("Enter scene description (e.g., 'a snowy forest with pine trees'): ")

    if ENABLE_META_LEARNING:
        print("\n🧠 Meta-learning is ENABLED - checking database for insights...")
        prompt = enhance_prompt_with_meta_learning(original_prompt)

        if prompt != original_prompt:
            print(f"\n🚀 Meta-learning enhanced your prompt!")
            print(f"   Original: '{original_prompt}'")
            print(f"   Enhanced: '{prompt}'")
        else:
            print(f"\n🚀 No similar scenes found (similarity < 0.5) - using original prompt")
            prompt = original_prompt
    else:
        print("\n⚠️  Meta-learning DISABLED by user setting")
        prompt = original_prompt
        print(f"   Using: '{prompt}'")

    import re
    # Sanitize folder name - remove all special characters
    safe_folder_name = re.sub(r'[^\w\s-]', '', original_prompt).strip()
    safe_folder_name = re.sub(r'[-\s]+', '_', safe_folder_name).lower()
    # Limit length to prevent path issues
    safe_folder_name = safe_folder_name[:100]

    project_dir = safe_folder_name
    assets_download_dir = os.path.join(project_dir, "assets")
    scene_output_dir = os.path.join(project_dir, "scene")
    os.makedirs(assets_download_dir, exist_ok=True)
    os.makedirs(scene_output_dir, exist_ok=True)

    # ✅ START TIMING
    pipeline_start_time = time.time()
    generation_id = None
    correction_count = 0
    final_success = False
    final_error_log = None

    print(f"Starting scene planning for prompt: '{prompt}'\n" + "-"*80)

    # STEP 1: Asset Identification
    print("Step 1: Asset Identification")
    asset_output_raw = get_agent_response(asset_identifier_agent, f"Identify 3-5 key 3D assets for a '{prompt}' scene.")
    asset_output_parsed = parse_json_safely(extract_content_from_response(asset_output_raw))
    print(safe_json_dumps(asset_output_parsed) + "\n" + "-"*80)

    # STEP 2: High-Level Planning
    print("Step 2: High-Level Planning")
    high_level_output_raw = get_agent_response(high_level_planner_agent, f"Create a high-level plan for a '{prompt}' scene with these assets: {safe_json_dumps(asset_output_parsed)}.")
    high_level_output_parsed = parse_json_safely(extract_content_from_response(high_level_output_raw))
    print(safe_json_dumps(high_level_output_parsed) + "\n" + "-"*80)

    # STEP 2.5: Spatial Analysis
    print("Step 2.5: Spatial Analysis")
    spatial_input = f"Analyze spatial relationships in this scene: '{prompt}'. Assets identified: {safe_json_dumps(asset_output_parsed)}"
    spatial_output_raw = get_agent_response(spatial_analyzer_agent, spatial_input)
    spatial_output_parsed = parse_json_safely(extract_content_from_response(spatial_output_raw))
    print(safe_json_dumps(spatial_output_parsed) + "\n" + "-"*80)

    # STEP 3: Asset Fetching
    print("Step 3: Asset Fetching")
    asset_list = asset_output_parsed if isinstance(asset_output_parsed, list) else []
    asset_fetcher_result = process_asset_list_sketchfab(asset_list, assets_download_dir)
    print("Asset fetching complete.\n" + "-"*80)

    # STEP 4: Asset Measurement
    print("Step 4: Asset Measurement")
    assets_to_measure = asset_fetcher_result.get("assets_mapping", {})
    if not assets_to_measure:
        print("No assets were found or downloaded. Aborting.")

        if ENABLE_DATABASE:
        # ✅ LOG FAILURE TO DATABASE (asset download failed)
            generation_time = time.time() - pipeline_start_time
            store_generation_result(
                prompt=prompt,
                success=False,
                project_path=project_dir,
                generation_time_seconds=generation_time,
                correction_attempts=0,
                final_attempt_number=0,
                assets_requested=len(asset_list),
                assets_found=0,
                assets_used=0,
                final_error_log="No assets could be downloaded from Sketchfab",
                error_type='asset_download_failed'
            )
        return

    measured_assets_result = measure_assets_in_blender(assets_to_measure, project_dir, BLENDER_EXECUTABLE_PATH)
    if not measured_assets_result:
        print("Aborting pipeline due to measurement failure.")

        # ✅ LOG FAILURE TO DATABASE (measurement failed)
        if ENABLE_DATABASE:
            generation_time = time.time() - pipeline_start_time
            store_generation_result(
                prompt=prompt,
                success=False,
                project_path=project_dir,
                generation_time_seconds=generation_time,
                correction_attempts=0,
                final_attempt_number=0,
                assets_requested=len(asset_list),
                assets_found=len(assets_to_measure),
                assets_used=0,
                final_error_log="Blender asset measurement process failed",
                error_type='measurement_failed'
            )
        return
    print(measured_assets_result)
    # STEP 5: Detailed Planning
    print("Step 5: Detailed Planning")
    detailed_input = (f"Create a detailed placement plan for a '{prompt}' scene.\n"
                    f"High-level plan: {safe_json_dumps(high_level_output_parsed)}\n"
                    f"Spatial constraints: {safe_json_dumps(spatial_output_parsed)}\n"
                    f"Assets with dimensions: {safe_json_dumps(measured_assets_result)}")

    detailed_output_raw = get_agent_response(detailed_planner_agent, detailed_input)
    detailed_output_parsed = parse_json_safely(extract_content_from_response(detailed_output_raw))
    print(safe_json_dumps(detailed_output_parsed) + "\n" + "-"*80)

    # --- STEP 6: CODE GENERATION WITH DATABASE TRACKING ---
    print("Step 6: Code Generation and Correction Loop")

    codegen_input = {
        "detailed_plan": detailed_output_parsed,
        "spatial_constraints": spatial_output_parsed,
        "fetched_assets": measured_assets_result.get("assets_mapping", {}),
        "asset_instructions": f"Use ONLY these exact assets: {list(measured_assets_result.get('assets_mapping', {}).keys())}",
        "original_prompt": prompt
    }
    codegen_prompt = json.dumps(codegen_input)

    max_correction_attempts = 10
    script_content = None
    final_script_path = os.path.join(scene_output_dir, "generate_scene.py")

    for attempt in range(max_correction_attempts):
        attempt_start_time = time.time()

        if attempt == 0:
            print(f"\n--- Attempt {attempt + 1}: Generating initial script ---")
            agent_to_call = code_generator_agent
            prompt_to_send = codegen_prompt
        else:
            print(f"\n--- Attempt {attempt + 1}: Correcting failed script ---")
            agent_to_call = code_corrector_agent

            # ✅ SIMPLE: Send full script + error
            prompt_to_send = f"""
    You must fix this Blender Python script that is failing.

    FULL ERROR OUTPUT:
    {error_output}

    COMPLETE SCRIPT TO FIX:
    ```python
    {script_content}
    ```

    Return the corrected script as JSON: {{"script": "fixed_code_here"}}

    CRITICAL REMINDERS:
    - Fix ONLY the error mentioned in the traceback
    - Do NOT change working code
    - Return the ENTIRE script with the fix applied
    - Use proper JSON escaping (\\n for newlines, \\\\ for backslashes)
    """
            correction_count += 1

        response_raw = get_agent_response(agent_to_call, prompt_to_send)
        response_parsed = parse_json_safely(extract_content_from_response(response_raw))

        if "script" not in response_parsed:
            print(f"✗ Agent '{agent_to_call.name}' failed to produce a valid script.")
            error_output = "Agent did not return a valid script."

            # ✅ CREATE GENERATION RECORD ON FIRST ATTEMPT
            if ENABLE_DATABASE:
                if generation_id is None:
                    generation_id = store_generation_result(
                        prompt=prompt,
                        success=False,
                        project_path=project_dir,
                        generation_time_seconds=0,
                        correction_attempts=0,
                        final_attempt_number=attempt + 1,
                        assets_requested=len(asset_list),
                        assets_found=len(measured_assets_result.get('assets_mapping', {})),
                        assets_used=0
                    )

                # Store this failed attempt
                attempt_time = time.time() - attempt_start_time
                store_correction_attempt(
                    generation_id=generation_id,
                    attempt_number=attempt + 1,
                    agent_used=agent_to_call.name,
                    error_occurred=True,
                    error_traceback=error_output,
                    execution_time_seconds=attempt_time
                )
            continue

        script_content = response_parsed["script"]
        error_output = run_blender_script(script_content, project_dir, BLENDER_EXECUTABLE_PATH)

        # ✅ CREATE GENERATION RECORD ON FIRST ATTEMPT
        if ENABLE_DATABASE:
            if generation_id is None:
                generation_id = store_generation_result(
                    prompt=prompt,
                    success=False,
                    project_path=project_dir,
                    generation_time_seconds=0,
                    correction_attempts=0,
                    final_attempt_number=attempt + 1,
                    assets_requested=len(asset_list),
                    assets_found=len(measured_assets_result.get('assets_mapping', {})),
                    assets_used=len(measured_assets_result.get('assets_mapping', {}))
                )

            # ✅ STORE EACH ATTEMPT IMMEDIATELY
            attempt_time = time.time() - attempt_start_time
            store_correction_attempt(
                generation_id=generation_id,
                attempt_number=attempt + 1,
                agent_used=agent_to_call.name,
                error_occurred=(error_output is not None),
                error_traceback=error_output,
                execution_time_seconds=attempt_time
            )

        if error_output is None:
            # ✅ SUCCESS CASE - Extract metrics and validate
            print("\n" + "="*80)
            print("SUCCESS: Script executed perfectly!")
            print(f"✓ Final Blender script saved to: {final_script_path}")
            print("="*80)

            scene_folder = os.path.join(project_dir, "scene")

            # Extract Blender metrics FIRST
            blender_metrics = extract_scene_metrics_from_blender(scene_folder, BLENDER_EXECUTABLE_PATH)

            # ✅ CRITICAL FIX: Safely extract with type validation
            def safe_extract(metrics_dict, key, default=None):
                """Extract value ensuring it's NOT a dict"""
                if not isinstance(metrics_dict, dict):
                    return default
                value = metrics_dict.get(key, default)
                # CRITICAL: If value is still a dict, return None
                if isinstance(value, dict):
                    print(f"⚠️ WARNING: {key} is a dict, returning None")
                    return default
                return value

            # Extract ALL metrics using safe extraction
            terrain_relief = safe_extract(blender_metrics, 'terrain_relief_m')
            terrain_rms = safe_extract(blender_metrics, 'terrain_rms_disp')
            mean_nnd = safe_extract(blender_metrics, 'mean_nnd_xy')
            global_density = safe_extract(blender_metrics, 'global_density_per_m2')
            out_of_bounds = safe_extract(blender_metrics, 'out_of_bounds_count')
            coverage_pct = safe_extract(blender_metrics, 'projected_coverage_pct')
            total_verts = safe_extract(blender_metrics, 'total_vertices')
            total_tris = safe_extract(blender_metrics, 'total_triangles')

            # DEBUG: Print what we extracted
            print(f"\n🔍 DEBUG - Extracted metrics:")
            print(f"  terrain_relief: {terrain_relief} (type: {type(terrain_relief)})")
            print(f"  terrain_rms: {terrain_rms} (type: {type(terrain_rms)})")
            print(f"  mean_nnd: {mean_nnd} (type: {type(mean_nnd)})")
            print(f"  global_density: {global_density} (type: {type(global_density)})")
            print(f"  out_of_bounds: {out_of_bounds} (type: {type(out_of_bounds)})")
            print(f"  coverage_pct: {coverage_pct} (type: {type(coverage_pct)})")
            print(f"  total_verts: {total_verts} (type: {type(total_verts)})")
            print(f"  total_tris: {total_tris} (type: {type(total_tris)})")

            # Run CLIP validation
            clip_scene_score = validate_scene_with_clip(
                prompt=original_prompt,
                scene_folder=scene_folder,
                clip_threshold=0.25
            )

            # ✅ UPDATE GENERATION RECORD WITH FINAL SUCCESS DATA
            if ENABLE_DATABASE:
                generation_time = time.time() - pipeline_start_time
                conn = psycopg2.connect(**DB_CONFIG)
                cur = conn.cursor()
                cur.execute("""
                    UPDATE generations SET
                        success = TRUE,
                        correction_attempts = %s,
                        generation_time_seconds = %s,
                        clip_scene_score = %s,
                        validation_passed = %s,
                        terrain_relief_m = %s,
                        terrain_rms_disp = %s,
                        mean_nnd_xy = %s,
                        global_density_per_m2 = %s,
                        out_of_bounds_count = %s,
                        projected_coverage_pct = %s,
                        total_vertices = %s,
                        total_triangles = %s
                    WHERE id = %s
                """, (
                    correction_count,
                    generation_time,
                    clip_scene_score,
                    (clip_scene_score >= 0.25) if clip_scene_score else False,
                    terrain_relief,
                    terrain_rms,
                    mean_nnd,
                    global_density,
                    out_of_bounds,
                    coverage_pct,
                    total_verts,
                    total_tris,
                    generation_id
                ))
                conn.commit()
                cur.close()
                conn.close()

                # ✅ STORE ASSET DETAILS
                store_asset_details(generation_id, measured_assets_result, detailed_output_parsed)

                # ✅ UPDATE ASSET PERFORMANCE
                for asset_name in measured_assets_result.get('assets_mapping', {}).keys():
                    clip_data = measured_assets_result.get('clip_thumbnails', {}).get(asset_name, {})
                    update_asset_performance(
                        asset_name=asset_name,
                        success=True,
                        clip_score=clip_data.get('clip_score')
                    )

                print(f"\n💾 All data saved to database (Generation ID: {generation_id})")
                print(f"🎯 CLIP Score: {clip_scene_score:.3f}")
                print(f"⏱️  Total Time: {generation_time:.1f}s")
                print(f"🔄 Corrections: {correction_count}")

            final_success = True
            return
        else:
            final_error_log = error_output
            print("\n--- Script failed. Waiting 65s before retry ---")
            time.sleep(65)

    # ✅ FAILURE CASE - Update generation record after max attempts
    if not final_success and ENABLE_DATABASE:
        generation_time = time.time() - pipeline_start_time
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            UPDATE generations SET
                success = FALSE,
                correction_attempts = %s,
                generation_time_seconds = %s,
                final_error_log = %s,
                error_type = 'max_attempts_exceeded'
            WHERE id = %s
        """, (correction_count, generation_time, final_error_log, generation_id))
        conn.commit()
        cur.close()
        conn.close()

        # ✅ UPDATE ASSET PERFORMANCE (mark as failure)
        for asset_name in measured_assets_result.get('assets_mapping', {}).keys():
            update_asset_performance(asset_name=asset_name, success=False)

        print(f"\n💾 Failure logged to database (Generation ID: {generation_id})")

    print(f"✗ FAILURE: Could not generate working script after {max_correction_attempts} attempts.")

if __name__ == "__main__":
    main()

