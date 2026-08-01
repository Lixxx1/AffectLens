from __future__ import annotations

import json

from src.metrics import EMOTION_BINDINGS, TASK_LABELS


SYSTEM_PROMPT = (
    "You are a careful visual affect annotator and art critic specializing in emotion perception from artworks. "
    "Classify the dominant perceived emotion of each artwork as a visual whole, using visually salient evidence before "
    "judging the overall atmosphere. Return only a valid JSON object."
)


def label_list(task: str) -> str:
    return ", ".join(TASK_LABELS[task])


def expected_output_schema() -> dict[str, str]:
    return {
        "dominant_emotion": "label from the list",
        "valence": "Positive or Negative",
        "arousal": "Low or High",
    }


def emotion_binding_text() -> str:
    lines = []
    for (arousal, valence), emotions in EMOTION_BINDINGS.items():
        lines.append(
            f"- arousal={arousal}, valence={valence}: "
            f"dominant_emotion must be one of {', '.join(emotions)}"
        )
    return "\n".join(lines)


def visual_reading_protocol() -> str:
    return (
        "Visual reading protocol:\n"
        "- First separate the artwork itself from external display, capture, and mounting artifacts.\n"
        "- Classify the dominant perceived emotion of the artwork as a visual whole, not only the literal emotion of a "
        "depicted person and not the artist's private intention.\n"
        "- Find the primary subject or focal relationship before judging mood. When a clear human or animal subject is present, "
        "weigh expressive posture and the main compositional focus together.\n"
        "- If there is no clear human or animal subject, do not force facial or bodily interpretation. For landscapes, still life, "
        "abstract, or decorative works, judge emotion from formal, atmospheric, symbolic, and material evidence.\n"
        "- Weight colors and background by visual salience and narrative role, not by pixel area alone. A large background should "
        "not decide the class merely because it covers most pixels, but if it carries the main atmosphere, pressure, emptiness, "
        "threat, calm, or loneliness, treat it as strong evidence.\n"
        "- For multiple figures or elements, classify the emotional impression led by the main subject or focal relationship. "
        "Do not let a secondary object or background element determine the whole artwork merely because it is emotionally obvious.\n"
        "- Actively interpret readable text inside the artwork or photo as thematic and emotional evidence, especially when it "
        "carries cultural, poetic, historical, or symbolic meaning.\n"
        "- However, do not determine the image emotion solely from the text. Use the text as important supporting evidence and "
        "combine it with visual evidence and overall visual tone. If the text and the visual mood reinforce "
        "each other, explain how they work together. If they seem to conflict, explain the tension and infer the final emotion "
        "from their combined meaning rather than simply copying the sentiment of the written words.\n"
    )


def artifact_calibration_rules() -> str:
    return (
        "Material and capture calibration:\n"
        "- Treat aging, preservation marks, lighting/camera effects, resolution limits, and display/mounting context as non-artwork "
        "evidence unless visibly integrated into the artwork's own content.\n"
    )


def expression_and_arousal_rules() -> str:
    return (
        "Expression and arousal calibration:\n"
        "- Do not map isolated cues such as tears, lowered gaze, solitude, or stillness directly to Sad; read them in context.\n"
        "- High arousal may come from coherent static tension or psychological pressure, not only motion; do not infer it from "
        "a single sharp line, contrast, or diagonal.\n"
        "- Low arousal means relaxed, softened, stagnant, heavy, tired, or stable energy, not merely the absence of large gestures.\n"
        "- Do not assign preset emotions to recurring motifs or culturally loaded symbols. Interpret them through their role in "
        "the whole artwork, genre context, visual salience, and relationship to surrounding evidence.\n"
    )


def emotion_calibration_rules() -> str:
    return (
        "Emotion label calibration:\n"
        "- Use Alarmed only for threat, danger, fear, panic, shock, acute tension, or an atmosphere of immediate risk. Choose it "
        "over Annoyed or Frustrated only when fear, danger, or alarm dominates.\n"
        "- Use Annoyed for irritation, displeasure, harshness, abrasive tension, tight facial resistance, or unpleasantness without "
        "fear and without a strong sense of blocked effort.\n"
        "- Use Frustrated for blocked energy, struggle, pressure, conflict, agitation, or effort that feels obstructed rather than "
        "simple irritation or fear.\n"
        "- Use Sad for loss, grief, loneliness, melancholy, emotional heaviness, drooping posture, or mournful atmosphere. Choose "
        "Sad over Tired or Bored when the low energy feels emotionally painful, lonely, or mournful.\n"
        "- Use Tired for exhaustion, drained energy, weariness, sleepiness, or depleted posture without clear grief or boredom.\n"
        "- Use Bored for dull, stagnant, flat, repetitive, disengaged, or emotionally under-stimulated impressions without grief "
        "and without physical exhaustion as the main cue.\n"
        "- Use Aroused for sensual, intimate, bodily, lush, or positively charged intensity; do not use it for generic excitement "
        "without sensual or bodily charge.\n"
        "- Use Excited for vivid, energetic, anticipatory, celebratory, or dynamic positive intensity. Choose it over Happy when "
        "high energy, momentum, or expectation dominates.\n"
        "- Use Happy for clear joy, delight, smiling warmth, or cheerful pleasure that is positive but less kinetic than Excited "
        "and less sensual than Aroused.\n"
        "- Use Glad for gentle cheerfulness, pleasant everyday joy, or light positive feeling. Glad is brighter than Calm but less "
        "settled or satisfied than Contentment.\n"
        "- Use Contentment for peaceful satisfaction, settled pleasure, comfort, ease, sufficiency, dwelling, leisure, appreciation, "
        "cultivated enjoyment, abundance, harvest, prosperity, social harmony, or being at home in a scene. Contentment is warmer and more "
        "satisfied than Calm, and steadier than Glad. Use these cues for Contentment when they present stable, low-arousal positive "
        "satisfaction rather than celebratory high energy. If a label seems like Content, return the canonical label Contentment.\n"
        "- Use Calm for soothing quietness, composure, balance, spaciousness, or neutral-positive low arousal without clear "
        "satisfaction, cheerfulness, grief, boredom, or exhaustion.\n"
    )


def inferred_style_skill_rules() -> str:
    return (
        "Style-aware skill use:\n"
        "- First inventory the visible elements, focal relationships, text/symbols, medium cues, and formal evidence.\n"
        "- Only after that inventory, infer likely art tradition, movement, medium, region, and period from visible cues.\n"
        "- Actively use any relevant installed art-history or art-style skills as interpretive aids; if several match, synthesize them.\n"
        "- If needed, and if a relevant skill is unclear, incomplete, or too general for this artwork, inspect appendix and search the sources.\n"
        "- Let skill guidance calibrate emotion cues, but do not let style stereotypes override the focal evidence from this artwork.\n"
        "- Choose labels from the whole artwork and the required valence-arousal binding.\n"
        "- In style_skill_analysis, briefly state the inferred style/tradition, the relevant skill guidance used, how it changed "
        "the emotion reading, any skill-provided sources checked when clarification was needed, and how you avoided relying on "
        "style stereotype alone. If no specific skill clearly applies, say so.\n"
    )


def reference_calibration_rules() -> str:
    return (
        "Reference use:\n"
        "- Image 1 is the only target. Images 2-5 are labeled references, NOT TARGET.\n"
        "- Read annotation.json for the correct labels of the attached references, matched by rank.\n"
        "- Use the labeled references to learn this dataset's annotation standard: evidence choice, description wording, "
        "emotion, valence, and arousal.\n"
        "- Let reference labels guide similar target evidence, but do not copy them unless the target supports the same reading.\n"
        "- Do not vote or average across references.\n"
        "- Final descriptions, reasoning, and labels must be only about Image 1.\n"
    )


def target_only_style_rules() -> str:
    return (
        "Style-aware calibration:\n"
        "- First inventory the visible elements, focal relationships, text/symbols, medium cues, and formal evidence.\n"
        "- Only after that inventory, infer a likely art tradition, movement, medium, region, or period from visible cues.\n"
        "- Use generally known style characteristics only as secondary calibration; do not claim that local skill files or sources were consulted.\n"
        "- Let style knowledge explain target evidence, but do not let a style stereotype override it.\n"
        "- In style_skill_analysis, set skills_applied to None (local skills unavailable) and describe any style-aware reading cautiously.\n"
    )


def target_only_reference_rules() -> str:
    return (
        "Reference use:\n"
        "- No retrieval references are available in this run.\n"
        "- Analyze only the attached target artwork and do not imply that reference images or annotation files were inspected.\n"
    )


def build_user_prompt() -> str:
    schema = json.dumps(expected_output_schema(), ensure_ascii=False)

    return (
        "Classify the dominant perceived emotion of this artwork as a visual whole based on the following labels. "
        "Do not reduce the task to a depicted person's expression or infer the artist's private intention. "
        "Apply the visual reading, calibration, and style-skill rules before choosing labels.\n\n"
        f"- dominant_emotion: {label_list('dominant_emotion')}\n"
        f"- valence: {label_list('valence')}\n"
        f"- arousal: {label_list('arousal')}\n\n"
        "The dominant_emotion label is hard-bound to valence and arousal. "
        "Use this binding as a known rule, not as a suggestion:\n"
        f"{emotion_binding_text()}\n\n"
        f"{visual_reading_protocol()}\n"
        f"{artifact_calibration_rules()}\n"
        f"{expression_and_arousal_rules()}\n"
        f"{emotion_calibration_rules()}\n"
        f"{inferred_style_skill_rules()}\n"
        f"{reference_calibration_rules()}\n"
        "Decision order:\n"
        "1. Remove external artifact evidence from consideration.\n"
        "2. Locate the primary subject or focal relationship; use formal evidence when faces or bodies are not central.\n"
        "3. Gather concrete artwork-internal evidence from color, composition, subject, symbols/text, brushwork, texture, light, and space.\n"
        "4. Infer likely style/tradition and use relevant installed skills only to calibrate the evidence.\n"
        "5. Judge valence and arousal from the whole visual structure, then choose dominant_emotion only from the matching quadrant.\n\n"
        "Return the result as a JSON object with this structure:\n"
        f"{schema}\n\n"
        "Do not include any reasoning, markdown, or extra text."
    )


def analysis_response_structure() -> dict[str, object]:
    return {
        "element_inventory": (
            "concise inventory of the visible artwork elements before style selection: main subjects or forms, focal relationship, "
            "text or symbols if present, medium/material cues, and key color/composition/line/light/brush evidence"
        ),
        "style_skill_analysis": {
            "inferred_style_or_tradition": "likely style, tradition, medium, region, or period inferred from visible evidence",
            "skills_applied": "relevant skills used and any skill-provided sources/appendix/references checked when needed, or None/Uncertain if no clear match",
            "skill_guided_reading": "how the skill guidance calibrated the emotional reading of this specific artwork",
            "style_bias_check": "how the final label avoids relying on a style stereotype alone",
        },
        "overall_caption": (
            "concise neutral caption of the artwork's visible content that closes by naturally conveying the overall mood "
            "or atmosphere, woven into the flow of the description itself rather than appended as a separate formulaic sentence"
        ),
        "brushstroke": "specific evidence from brushwork, paint handling, mark texture, or surface treatment, plus the perceptual or emotional effect it creates",
        "composition": "specific evidence from arrangement, focal structure, balance, cropping, movement, and spatial organization, plus the perceptual or emotional effect it creates",
        "color": "specific evidence from palette, saturation, value contrast, temperature, and color relationships, plus the perceptual or emotional effect it creates",
        "line": "specific evidence from contours, edges, linear direction, curvature, sharpness, and line tension, plus the perceptual or emotional effect it creates",
        "light": "specific evidence from illumination, shadow, highlights, darkness, and light quality, plus the perceptual or emotional effect it creates",
        "reasoning": (
            "concise evidence synthesis after visual evidence and style-skill calibration: filter external artifacts, "
            "identify the focal evidence, decide valence/arousal, and explain why the final emotion is best"
        ),
        "valence_reasoning": "why the image is Positive or Negative",
        "arousal_reasoning": "why the image is Low or High arousal",
        "emotion_reasoning": "why the selected dominant_emotion is the best label",
        "quadrant_check": "explain that the emotion is consistent with the selected valence/arousal quadrant",
        "second_best_emotion": "another plausible label from the list",
        "dominant_emotion": "label from the list",
        "valence": "Positive or Negative",
        "arousal": "Low or High",
        "uncertainty": "Low, Medium, or High",
    }


def build_analysis_prompt(
    *,
    include_references: bool = True,
    include_local_skills: bool = True,
) -> str:
    schema = json.dumps(analysis_response_structure(), ensure_ascii=False)
    style_rules = (
        inferred_style_skill_rules()
        if include_local_skills
        else target_only_style_rules()
    )
    reference_rules = (
        reference_calibration_rules()
        if include_references
        else target_only_reference_rules()
    )
    style_step = (
        "3. Use that element inventory to infer likely style/tradition and select any relevant installed skills; apply skills only as calibration.\n"
        if include_local_skills
        else "3. Use that element inventory to infer likely style/tradition; apply general style knowledge only as cautious secondary calibration.\n"
    )
    reference_step = (
        "4. After completing style_skill_analysis, inspect the current "
        "folder's query image, rank01...rank10 similar reference images with similarity scores, filenames, and annotation.json. "
        "Attached reference images are reference only. Do not treat reference-image content as the target artwork.\n"
        if include_references
        else "4. No retrieval references are available; continue using only target-artwork evidence.\n"
    )
    drafting_step = (
        "5. Only after the element inventory, style-skill calibration, and folder-reference calibration, begin drafting the output fields.\n"
        if include_references
        else "5. Only after the element inventory and style-aware calibration, begin drafting the output fields.\n"
    )
    bias_recheck = (
        "8. Recheck whether secondary elements, text, style stereotypes, external artifacts, or reference images biased the judgment; correct if needed.\n\n"
        if include_references
        else "8. Recheck whether secondary elements, text, style stereotypes, or external artifacts biased the judgment; correct if needed.\n\n"
    )
    final_order_rule = (
        "Do not decide valence, arousal, or dominant_emotion before completing style_skill_analysis and folder-reference calibration. "
        if include_references
        else "Do not decide valence, arousal, or dominant_emotion before completing style_skill_analysis. "
    )

    return (
        "Classify the dominant perceived emotion of this artwork as a visual whole based on artwork-internal evidence only, including readable text inside the artwork when present. "
        "Do not reduce the task to a depicted person's expression or infer the artist's private intention. Judge by visual salience "
        "and narrative role rather than treating the whole image as an equal-area color map.\n\n"
        "The dominant_emotion label is hard-bound to valence and arousal. "
        "Use this binding as a known rule, not as a suggestion:\n"
        f"{emotion_binding_text()}\n\n"
        f"{visual_reading_protocol()}\n"
        f"{artifact_calibration_rules()}\n"
        f"{expression_and_arousal_rules()}\n"
        f"{emotion_calibration_rules()}\n"
        f"{style_rules}\n"
        f"{reference_rules}\n"
        "Decide in this order:\n"
        "1. Exclude external artifact evidence.\n"
        "2. Build an element inventory: primary subject or focal relationship, important objects, figures, text/symbols, medium cues, "
        "and concrete formal evidence from color, composition, brushstroke, line, light, texture, and space.\n"
        f"{style_step}"
        f"{reference_step}"
        f"{drafting_step}"
        "6. Decide valence and arousal from the whole visual structure of the target artwork.\n"
        "7. Choose dominant_emotion only from the matching quadrant, then compare it with second_best_emotion.\n"
        f"{bias_recheck}"
        "When writing the six result description fields, keep their meanings separate: "
        "overall_caption is a neutral whole-image caption that ends by conveying the overall mood or atmosphere naturally within the description, not as a separate stock sentence, brushstroke is mark/surface handling, composition is arrangement, "
        "color is palette and contrast, line is contour/edge/directional tension, and light is illumination/shadow quality. "
        "For brushstroke, composition, color, line, and light, state both the concrete artwork-internal evidence and the "
        "perceptual or emotional effect caused by that evidence, such as pressure, calm, heaviness, warmth, distance, tension, "
        "intimacy, instability, or release. Keep the effect tied to that field's evidence rather than repeating the final label. "
        "Do not duplicate brushstroke into line.\n\n"
        "Return a valid JSON object with this structure:\n"
        f"{schema}\n\n"
        "Write fields in the schema order: element_inventory first, style_skill_analysis second, then the six result description fields, "
        "then reasoning and the later label reasoning fields. "
        f"{final_order_rule}"
        "Keep each explanation concise. Do not include markdown or extra text outside JSON."
    )
