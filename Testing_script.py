import os
import re
import json
import time
import glob
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
from PIL import Image
from openai import OpenAI

IMGBB_API_KEYS = [
]

client = OpenAI(
    base_url="",
    api_key=""
)

def setup_logger():
    log_dir = Path("./Test/Evaluation_Logs")
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = log_dir / f"image2interleave_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

def upload_to_imgbb(file_path: Path) -> Optional[str]:
    for api_key in IMGBB_API_KEYS:
        try:
            with open(file_path, "rb") as f:
                m = MultipartEncoder(fields={
                    "key": api_key,
                    "image": (file_path.name, f, "image/jpeg")
                })
                response = requests.post(
                    "https://api.imgbb.com/1/upload",
                    data=m,
                    headers={"Content-Type": m.content_type},
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                if data.get("success"):
                    logger.info(f"上传成功: {data['data']['url']}")
                    return data["data"]["url"]
        except Exception as e:
            logger.warning(f"上传失败 (key {api_key[:5]}...): {e}")
            time.sleep(1)
    
    logger.error("所有API密钥上传均失败")
    return None

def load_samples_from_folder(folder_path: str) -> List[Dict]:
    pattern = os.path.join(folder_path, "sample*_result.json")
    json_files = sorted(
        glob.glob(pattern),
        key=lambda x: int(re.search(r'sample(\d+)_result', x).group(1))
    )
    
    samples = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                samples.append(json.load(f))
        except Exception as e:
            logger.error(f"无法读取 {json_file}: {e}")
    
    logger.info(f"加载了 {len(samples)} 个样本")
    return samples

def create_image_grid_with_reference(gen_images: List[str], gt_image: str, output_path: Path) -> bool:
    try:
        images_to_grid = []
        for i in range(1, 6):  # gen_2(idx=1) 到 gen_6(idx=5)
            if i < len(gen_images):
                images_to_grid.append(Image.open(gen_images[i]).convert('RGB'))
            else:
                logger.error(f"缺少 gen_{i+1}")
                # return False

        images_to_grid.append(Image.open(gt_image).convert('RGB'))

        target_size = images_to_grid[0].size
        resized_images = [img.resize(target_size, Image.Resampling.LANCZOS) for img in images_to_grid]
        rows, cols = 2, 3
        canvas_width = cols * target_size[0]
        canvas_height = rows * target_size[1]
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')

        for idx, img in enumerate(resized_images):
            row = idx // cols
            col = idx % cols
            x = col * target_size[0]
            y = row * target_size[1]
            canvas.paste(img, (x, y))
        
        canvas.save(output_path, quality=100)
        logger.info(f"拼图创建成功: 2x3 grid [gen2, gen3, gen4, gen5, gen6, gt]")
        return True
    except Exception as e:
        logger.error(f"拼图失败: {e}")
        return False

def extract_images_and_steps(sample_data: Dict, image_folder: str) -> Tuple[List[str], str, str]:
    sample_id = sample_data.get("sample_id")
    thinking = sample_data.get("text_description", "")
    
    if not thinking:
        logger.warning(f"Sample {sample_id}: 没有 'thinking' 字段")
        return [], "", ""
    
    gen_images = []
    for i in range(1, 7):  # 1-6
        img_path = Path(image_folder) / f"sample{sample_id}_gen_{i}.png"
        if img_path.exists():
            gen_images.append(str(img_path))
        else:
            logger.warning(f"Sample {sample_id}: 图片不存在 {img_path.name}")

    gt_path = Path(image_folder) / f"sample{sample_id}_gt.png"
    if not gt_path.exists():
        logger.error(f"Sample {sample_id}: GT图片不存在 {gt_path.name}")
        return [], "", ""
    
    logger.info(f"Sample {sample_id}: 提取 {len(gen_images)} 张生成图 + 1张GT图")
    return gen_images, str(gt_path), thinking


def evaluate_image2interleave(sample_data: Dict, gen_images: List[str], gt_image: str, thinking: str, collage_url: Optional[str]) -> Optional[Dict]:

    sample_id = sample_data.get("sample_id")
    instruction = sample_data.get("prompt", "")

    image_info = f"""
        **Generated Images:** {len(gen_images)} images total (gen_2 to gen_6)
        - Using for evaluation: gen_2, gen_3, gen_4, gen_5, gen_6 (5 images displayed in grid)
        - gen_1 omitted (initial/starting frame)

        **Reference Image (GT):** sample{sample_id}_gt.png
        - Position: Bottom-Center corner in the 2x3 grid
        - This is the REFERENCE IMAGE that the sequence should reflect/reconstruct

        **Grid Layout (2x3):**
        ┌─────────┬─────────┬─────────┐
        │  gen_2 │  gen_3  │  gen_4 │  ← Top row (Steps 2-4)
        ├─────────┼─────────┼─────────┤
        │  gen_5 │   gen6  │   GT   │  ← Bottom row (Steps 5-6 + Reference)
        └─────────┴─────────┴─────────┘

        **Reading Order:** LEFT-TO-RIGHT, TOP-TO-BOTTOM
        Position 1 (Top-Left): gen_2
        Position 2 (Top-Center): gen_3
        Position 3 (Top-Right): gen_4
        Position 4 (Bottom-Left): gen_5
        Position 5 (Bottom-Center): gen_6 (IF HAVE)
        Position 6 (Bottom-Right): GT
    """
    
    prompt = f"""You are an expert evaluator for IMAGE-CONDITIONED multimodal generation systems.

        EVALUATION TASK: Image2Interleave

        **Scenario:** Given a REFERENCE IMAGE (GT), the system generates a step-by-step tutorial sequence that:
        1. Reflects/reconstructs the reference image through progressive steps
        2. Provides textual descriptions for each step
        3. Maintains visual and semantic consistency

        **Original Instruction:**
        "{instruction}"

        {image_info}

        **Textual Descriptions (9 steps in thinking, 6 generated images but showing 5):**
        {thinking}

        **Note:** The 9-step description should logically map to the 6 generated images (we show 5: gen_2 through gen_6). Some textual steps may be combined or implicit in the images.

        {'**VISUAL REFERENCE PROVIDED:**' if collage_url else '** TEXT-ONLY EVALUATION:**'}
        {'''A 2x3 collage shows:
        Top Row: gen_2, gen_3, gen_4 (Steps 2-4)
        Bottom Row: gen_5, gen_6, GT (Steps 5-6 + REFERENCE)

        The GT image (bottom-right) is the TARGET/REFERENCE that the sequence aims to reflect.''' if collage_url else 'No visual reference. Evaluate based on text only.'}

        EVALUATION CRITERIA (1-5 Scale, STRICT)
        1️⃣ REFERENCE FAITHFULNESS (参考图忠实度) [1-5]
        **Core Question:** Do gen_2 through gen_6 ACCURATELY reflect/reconstruct the reference image (GT)?

        **Verification Checklist:**
        □ **Global Structure:** Do generated images share the same scene/subject as GT?
        □ **Key Objects:** Are main objects from GT visible across the 5 frames?
        □ **Visual Details:** Are colors, textures, compositions consistent with GT?
        □ **Local Fidelity:** Are fine details (e.g., ingredients, tools, arrangement) preserved?
        □ **Reconstruction Pathway:** Can you trace elements from GT through gen_2→gen_6?

        **Scoring Standards:**
        - **5 (Perfect):** GT image elements clearly visible and preserved across ALL 5 frames, perfect reconstruction pathway
        - **4 (Good):** Most GT elements present in 4/5 frames, minor details lost, but clear connection to reference
        - **3 (Acceptable):** Key GT elements recognizable in 3/5 frames, but significant detail loss or style drift
        - **2 (Poor):** GT barely recognizable in 1-2 frames, major elements missing or distorted
        - **1 (Fail):** No visible connection to GT in any frame, completely different scene/subject

        **AUTO-PENALTY:**
        - Key object from GT missing in generated images: -1 point
        - Color/style inconsistency with GT: -1 point
        - Completely different subject matter: Auto score = 1

        2️⃣ TEMPORAL COHERENCE (时序连贯性) [1-5]
        **Core Question:** Do gen_2→gen_3→gen_4→gen_5→gen_6 show smooth, logical progression TOWARD the GT state?

        **Verification Checklist:**
        □ **Directional Flow:** Are steps moving TOWARD the GT state (not random)?
        □ **Logical Sequence:** Does each frame build upon the previous one?
        □ **Plausible Transitions:** Are changes between consecutive frames realistic?
        □ **No Temporal Jumps:** Are there missing critical steps between frames?
        □ **Reference Convergence:** Does the sequence logically lead to the GT image?

        **Frame-by-Frame Check:**
        - gen_2 → gen_3: Smooth transition? ✓/✗
        - gen_3 → gen_4: Logical progression? ✓/✗
        - gen_4 → gen_5: No jumps? ✓/✗
        - gen_5 → gen_6: Converging to GT? ✓/✗
        - gen_6 → GT: Natural endpoint? ✓/✗

        **Scoring Standards:**
        - **5 (Perfect):** All 4 transitions smooth, clear progression toward GT, no violations (5/5 ✓)
        - **4 (Good):** 1 minor transition issue, but overall direction toward GT clear (4/5 ✓)
        - **3 (Acceptable):** 2 transition issues or 1 major gap, but followable (3/5 ✓)
        - **2 (Poor):** 3+ issues, choppy sequence, unclear direction (≤2/5 ✓)
        - **1 (Fail):** Random frames, no progression, contradicts GT state (0/5 ✓)

        **⚠️ AUTO-PENALTY:**
        - Frames go in wrong direction (away from GT): -1 point per violation
        - Missing critical intermediate step: -1 point
        - Temporal loop or regression: -2 points


        3️⃣ SEMANTIC ALIGNMENT (语义对齐度) [1-5]
        **Core Question:** Do the 9 textual descriptions ACCURATELY map to the 6 images (especially gen_2-6) AND reference the GT?

        **Verification Checklist:**
        □ **Text-Image Match:** Does each description match its corresponding generated image?
        □ **Step Mapping:** Can you reasonably map 9 text steps to 6 images (5 shown)?
        □ **GT Acknowledgment:** Do descriptions acknowledge/reference the GT target state?
        □ **Action Accuracy:** Do described actions match visual changes between frames?
        □ **Completeness:** Does the text guide toward achieving the GT result?

        **Mapping Logic (9 steps → 6 images):**
        Possible groupings:
        - Steps 1-2 → gen_1 (omitted from display but may be referenced)
        - Step 3 → gen_2 (shown)
        - Step 4 → gen_3 (shown)
        - Steps 5-6 → gen_4 (shown)
        - Step 7 → gen_5 (shown)
        - Steps 8-9 → gen_6 (shown) + GT as final reference

        **Scoring Standards:**
        - **5 (Perfect):** ALL 9 descriptions clearly map to 6 images, perfect text-image match, GT well-referenced
        - **4 (Good):** Clear mapping with 1-2 minor ambiguities, GT referenced
        - **3 (Acceptable):** Reasonable mapping but 3-4 misalignments, GT mentioned
        - **2 (Poor):** Unclear mapping, 5+ misalignments, GT poorly referenced
        - **1 (Fail):** No logical mapping, text unrelated to images or GT

        **AUTO-PENALTY:**
        - Each clear text-image mismatch: -1 point
        - Text completely ignores GT reference: -2 points
        - Impossible mapping (can't logically group 9→6): Start from 2 max

        PRACTICAL USABILITY ASSESSMENT

        **Critical Questions:**
        1. Can someone USE this sequence to recreate the GT image/result?
        2. Are the 5 displayed steps (gen_2-6) actionable and clear?
        3. Is the image-text combination helpful or confusing?
        4. Would following this tutorial lead to the GT outcome in real life?
        5. Are the right moments/steps shown in the 5 frames?

        OUTPUT FORMAT
        
        Respond with ONLY this JSON (no markdown, no extra text):

        {{
            "reference_faithfulness": <integer 1-5>,
            "temporal_coherence": <integer 1-5>,
            "semantic_alignment": <integer 1-5>,
            
            "reference_reasoning": "<2-4 sentences: How well do gen_2-6 reflect GT? List preserved/missing key elements.>",
            "temporal_reasoning": "<2-4 sentences: Evaluate gen_2→gen_3→gen_4→gen_5→gen_6 transitions. Smooth progression toward GT?>",
            "semantic_reasoning": "<2-4 sentences: How do 9 text steps map to 6 images (5 shown)? Do descriptions match visuals? GT referenced?>",
            
            "overall_score": <float: average of 3 scores, 1 decimal>,
            
            "gt_reconstruction_quality": "<Excellent/Good/Fair/Poor - can you trace GT elements through gen_2→gen_6?>",
            "transition_quality": "<5/5, 4/5, 3/5, 2/5, 1/5, 0/5 - how many frame transitions are smooth?>",
            "text_image_mapping": "<Clear/Moderate/Unclear - how well do 9 steps map to 6 images?>",
            "usability_assessment": "<1-2 sentences: Can someone follow gen_2-6 to achieve the GT result?>",
            
            "detailed_alignment_check": {{
                "image_2": "<✓ Match which sentence / ✗ Mismatch: reason>",
                "image_3": "<✓ Match which sentence / ✗ Mismatch: reason>",
                "image_4": "<✓ Match which sentence / ✗ Mismatch: reason>",
                "image_5": "<✓ Match which sentence / ✗ Mismatch: reason>",
                "image_6": "<✓ Match which sentence / ✗ Mismatch: reason>"
            }},
            "total_image_text_alignment_count": <?/5>
        }}

        CRITICAL REMINDERS
        1. **GT IS THE ANCHOR:** All 5 frames (gen_2-6) must connect to the GT reference
        2. **CHECK ALL 5 TRANSITIONS:** gen_2→3, 3→4, 4→5, 5→6, and conceptual 6→GT
        3. **VERIFY 9→6 MAPPING:** Text steps must logically group to match 6 images
        4. **STRICT ON FAITHFULNESS:** Missing GT elements = low score
        5. **PRACTICAL LENS:** Would this tutorial actually work to recreate GT?

        The 2x3 grid layout is:
        [gen_2][gen_3][gen_4]
        [gen_5][gen_6][ GT  ]

        Now evaluate this Image2Interleave sequence:
    """

    messages = [
        {
            "role": "system",
            "content": "You are a world-class evaluator for image-conditioned multimodal generation. You specialize in assessing reference image faithfulness, temporal coherence, and text-image semantic alignment. Always respond with valid JSON only."
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]
    
    if collage_url:
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": collage_url,
                "detail": "high"
            }
        })
    
    for retry in range(3):
        try:
            logger.info(f"Sample {sample_id}: GPT-4评估中 (重试 {retry+1}/3)...")
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1800,
                temperature=0.2
            )
            
            content = response.choices[0].message.content.strip()

            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            scores = json.loads(content)
            required = [
                "reference_faithfulness",
                "temporal_coherence",
                "semantic_alignment"
            ]
            
            for field in required:
                if field not in scores:
                    raise ValueError(f"缺少必需字段: {field}")
                if not (1 <= scores[field] <= 5):
                    raise ValueError(f"字段 {field} 分数超出范围: {scores[field]}")

            if "overall_score" not in scores:
                scores["overall_score"] = round(
                    (scores["reference_faithfulness"] +
                     scores["temporal_coherence"] +
                     scores["semantic_alignment"]) / 3,
                    1
                )
            
            logger.info(f"Sample {sample_id}: 评估成功")
            logger.info(f"  - Reference Faithfulness: {scores['reference_faithfulness']}/5")
            logger.info(f"  - Temporal Coherence: {scores['temporal_coherence']}/5")
            logger.info(f"  - Semantic Alignment: {scores['semantic_alignment']}/5")
            logger.info(f"  - Overall: {scores['overall_score']:.1f}/5")
            logger.info(f"  - GT Reconstruction: {scores.get('gt_reconstruction_quality', 'N/A')}")
            
            return scores
            
        except json.JSONDecodeError as e:
            logger.error(f"Sample {sample_id}: JSON解析失败 - {e}")
            logger.error(f"原始响应前500字符: {content[:500]}")
            time.sleep(3)
        except Exception as e:
            logger.error(f"Sample {sample_id}: 评估失败 - {e}")
            time.sleep(5)
    
    logger.error(f"Sample {sample_id}: 所有重试失败")
    return None

def process_sample(sample_data: Dict, image_folder: str, output_dir: Path, enable_visual: bool) -> Optional[Dict]:
    # 处理单个样本
    sample_id = sample_data.get("sample_id")
    logger.info(f"\n{'='*60}\n处理 Sample {sample_id} {'='*60}")
    
    try:
        gen_images, gt_image, thinking = extract_images_and_steps(sample_data, image_folder)
        
        if not gen_images or not gt_image or not thinking:
            logger.warning(f"Sample {sample_id}: 跳过 (缺少数据)")
            return None
        
        collage_url = None
        if enable_visual:
            collage_path = output_dir / f"sample{sample_id}_collage.jpg"
            if create_image_grid_with_reference(gen_images, gt_image, collage_path):
                collage_url = upload_to_imgbb(collage_path)

        scores = evaluate_image2interleave(sample_data, gen_images, gt_image, thinking, collage_url)
        if not scores:
            return None

        return {
            "sample_id": sample_id,
            "prompt": sample_data.get("prompt", ""),
            "num_gen_images": len(gen_images),
            "has_gt": bool(gt_image),
            "collage_url": collage_url,
            "scores": scores,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Sample {sample_id}: 处理失败 - {e}", exc_info=True)
        return None

def main():
    parser = argparse.ArgumentParser(description='Image2Interleave 评估工具')
    parser.add_argument('--folder', default='', help='sample*_result.json文件夹')
    parser.add_argument('--image_folder', default='', help='图片文件夹')
    parser.add_argument('--output_dir', default='', help='输出目录')
    parser.add_argument('--enable_visual', default=True, help='启用拼图+上传')
    parser.add_argument('--start_from', type=int, default=0, help='起始sample_id')
    parser.add_argument('--limit', type=int, default=100, help='最大处理数量')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples = load_samples_from_folder(args.folder)
    if not samples:
        logger.error("没有加载到任何样本")
        return
    
    logger.info(f"\n{'='*60}Image2Interleave 评估启动{'='*60}")
    logger.info(f"样本总数: {len(samples)}")
    logger.info(f"起始ID: {args.start_from}")
    logger.info(f"限制数量: {args.limit or '无'}")
    logger.info(f"视觉评估: {'启用' if args.enable_visual else '禁用'}")
    
    results = []
    failed = []
    
    for sample in samples:
        sample_id = sample.get("sample_id")
        
        if sample_id < args.start_from:
            continue
        if args.limit and len(results) >= args.limit:
            logger.info(f"达到处理限制 ({args.limit})")
            break
        
        logger.info(f"进度: {len(results)+1}/{min(len(samples), args.limit or len(samples))} (成功: {len(results)})")
        result = process_sample(sample, args.image_folder, output_dir, args.enable_visual)
        
        if result:
            results.append(result)
            with open(output_dir / "evaluation_results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            failed.append(sample_id)
        
        time.sleep(2)

    if results:
        avg = {
            "reference_faithfulness": sum(r["scores"]["reference_faithfulness"] for r in results) / len(results),
            "temporal_coherence": sum(r["scores"]["temporal_coherence"] for r in results) / len(results),
            "semantic_alignment": sum(r["scores"]["semantic_alignment"] for r in results) / len(results),
            "overall_score": sum(r["scores"]["overall_score"] for r in results) / len(results)
        }
        
        summary = {
            "metadata": {
                "evaluation_type": "Image2Interleave",
                "total_samples": len(samples),
                "evaluated": len(results),
                "failed": len(failed),
                "failed_ids": failed,
                "timestamp": datetime.now().isoformat()
            },
            "average_scores": avg
        }
        
        with open(output_dir / "evaluation_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n{'='*60}评估完成{'='*60}")
        logger.info(f"成功: {len(results)}/{len(samples)}")
        logger.info(f"失败: {failed}")
        logger.info(f"\n平均分数:")
        logger.info(f"  Reference Faithfulness: {avg['reference_faithfulness']:.2f}/5.0")
        logger.info(f"  Temporal Coherence:     {avg['temporal_coherence']:.2f}/5.0")
        logger.info(f"  Semantic Alignment:     {avg['semantic_alignment']:.2f}/5.0")
        logger.info(f"  Overall Score:          {avg['overall_score']:.2f}/5.0")
        logger.info(f"\n输出: {output_dir}")
        logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()
