SYSTEM_PROMPT = """You are a helpful AI agent, a part of knowledge bridge 2, aimed at generating {target} details based on the information provided by the user. 

[JOB]
Your job is to understand the information provided by the user and imagine the potential missing modality.
"""

GENERATION_TEXT_PROMPT = """
[Context]
You are provided with multimodal observations that include an image and an audio clip. 

[Image Description]
{image_info}.

[Audio Description]
{audio_info}.

[Text Guess]
{guess}

[Task]
Based on the provided details, generate a series distinct candidate text narratives. Each narrative should:
1. Integrate both the visual and audio information seamlessly.
2. Highlight the interplay between the image and audio.
3. Present different stylistic or emotive perspectives.
4. **Contain only one object and one action.** Ensure that each candidate is a single, concise sentence that mentions exactly one object and one corresponding action.


[Output Format]
List the candidate texts as follows:
```json
{{
    "candidates": [
        {{ "text": "<candidate_1>" }},
        {{ "text": "<candidate_2>" }},
        ...
    ]
}}
```
"""

GENERATION_TEXT_PROMPT_WITHOUT_GRAPH = """
[Context]
You are provided with multimodal observations that include an image and an audio clip. 

[Image Description]
{image_info}.

[Audio Description]
{audio_info}.

**NOTE**: The audio may not be contained any useful information for the text generation. In this case, you can pay more attention to the image information.


[Task]
Based on the provided details, generate a series distinct candidate texts. Each text should:
1. Integrate both the visual and audio information seamlessly.
2. Highlight the interplay between the image and audio.
3. Present different stylistic or emotive perspectives.
4. **Contain only one object and one action.** Ensure that each candidate is a single, concise sentence that mentions exactly one object and one corresponding action.
5. Provide more details about the object and action in the candidate text.

[Output Format]
List the candidate texts as follows:
```json
{{
    "candidates": [
        {{ "text": "<candidate_1>" }},
        {{ "text": "<candidate_2>" }},
        ...
    ]
}}
```
"""


GENERATION_IMAGE_PROMPT = """
[Context]
You are provided with multimodal observations that include a text and an audio clip. 

[Text Description]
{text_info}.

[Audio Description]
{audio_info}.

[Image Guess]
{guess}

[Task]
Based on the provided information, generate a series distinct candidate prompts for image generation. Each prompt should:
1. Detail the visual elements (scene, characters, colors, style, etc.) informed by the audio and text.
2. Clearly bridge the auditory and narrative information to suggest a compelling visual outcome.
3. Offer varied creative interpretations for potential image outputs.
4. Ensure that the prompts at less 150 characters in length.
5. **Contain only one object and one action.** Ensure that each candidate is a single, concise sentence that mentions exactly one object and one corresponding action.


[Output Format]
List the candidate prompts as follows:
```json
{{
    "candidates": [
        {{ "prompts": "<candidate_1>" }},
        {{ "prompts": "<candidate_2>" }},
        ...
    ]
}}
```
"""

GENERATION_IMAGE_PROMPT_WITHOUT_GRAPH = """
[Context]
You are provided with multimodal observations that include a text and an audio clip. 

[Text Description]
{text_info}.

[Audio Description]
{audio_info}.

**NOTE**: The audio may not be contained any useful information for the image generation. In this case, you can pay more attention to the text information.

[Task]
Based on the provided information, generate a series distinct candidate prompts for image generation. Each prompt should:
1. Detail the visual elements (scene, characters, colors, style, etc.) informed by the audio and text.
2. Clearly bridge the auditory and narrative information to suggest a compelling visual outcome.
3. Offer varied creative interpretations for potential image outputs.
4. Ensure that the prompts at less 150 characters in length.
5. **Contain only one object and one action.** Ensure that each candidate is a single, concise sentence that mentions exactly one object and one corresponding action.
6. Provide more details about the object and action in the candidate prompt.

[Output Format]
List the candidate prompts as follows:
```json
{{
    "candidates": [
        {{ "prompts": "<candidate_1>" }},
        {{ "prompts": "<candidate_2>" }},
        ...
    ]
}}
```
"""

GENERATION_AUDIO_PROMPT = """
[Context]
You are provided with multimodal observations that include an image and a text. 

[Image Description]
{image_info}.

[Text Description]
{text_info}.

[Audio Guess]
{guess}

[Task]
Using the above information, generate a series distinct candidate prompts for synthesizing speech. Each candidate prompt should:
1. Integrate both the image and text details to guide the audio generation.
2. Clearly indicate the type of sound source:
   - **For non-human sounds (e.g., from an animal or object):** Begin the prompt with "the sound of [object]" where [object] is the single object producing the sound.
   - **For human speech:** Begin the prompt with "a man or a woman speak [action]" where [action] describes the single action (for example, a greeting, a remark, or a statement).
4. Use diverse styles in the candidate prompts while following the above format.
5. **Contain only one object and one action.** Ensure that each candidate is a single, concise sentence that mentions exactly one object and one corresponding action.
6. Ensure that the prompts at less 150 characters in length.


[Output Format]
List the candidate prompts as follows:
```json
{{
    "candidates": [
        {{ "prompts": "<candidate_1>" }},
        {{ "prompts": "<candidate_2>" }},
        ...
    ]
}}"""


GENERATION_AUDIO_PROMPT_WITHOUT_GRAPH = """
[Context]
You are provided with multimodal observations that include an image and a text. 

[Image Description]
{image_info}.

[Text Description]
{text_info}.


[Task]
Using the above information, generate a series distinct candidate prompts for synthesizing speech. Each candidate prompt should:
1. Integrate both the image and text details to guide the audio generation.
2. Clearly indicate the type of sound source:
   - **For non-human sounds (e.g., from an animal or object):** Begin the prompt with "the sound of [object]" where [object] is the single object producing the sound.
   - **For human speech:** Begin the prompt with "a man or a woman speak [action]" where [action] describes the single action (for example, a greeting, a remark, or a statement).
4. Use diverse styles in the candidate prompts while following the above format.
5. **Contain only one object and one action.** Ensure that each candidate is a single, concise sentence that mentions exactly one object and one corresponding action.
6. Ensure that the prompts at less 150 characters in length.


[Output Format]
List the candidate prompts as follows:
```json
{{
    "candidates": [
        {{ "prompts": "<candidate_1>" }},
        {{ "prompts": "<candidate_2>" }},
        ...
    ]
}}
"""


########## REFINED PROMPT ##########
GENERATION_IMAGE_PROMPT_REFINED = """[Context]
You are an expert image generation agent.  Your job is to take an existing prompt and its reviewer feedback, then produce a single, improved prompt that:

1. Keeps all the artistically valuable elements from the **Original Prompt**.  
2. Applies or corrects details according to the **Feedback**.  
3. Enhances clarity, composition, and style to better achieve the user's vision.

[Inputs]
Original Prompt:
"{original_prompt}"

Feedback:
"{feedback}"

[Instructions]
1. Read the Original Prompt and identify its core elements (subject, setting, style, mood, color palette, perspective, etc.).
2. Read each piece of Feedback and note:
   - Which details to **add** (e.g. "more dramatic lighting," "include a solitary tree silhouette").
   - Which details to **remove** or **tone down** (e.g. "avoid busy backgrounds," "less saturated colors").
   - Any stylistic shifts requested (e.g. "make it more atmospheric," "lean toward a painterly look").
3. Merge these into one **concise**, **coherent** prompt that:
   - Clearly enumerates the **final** composition instructions.
   - Specifies style, mood, and technical details (angle, lighting, color scheme, resolution hints).
   - Uses vivid, image-generation-friendly language.
4. Ensure that the prompts at less 150 characters in length.

[Output]
Produce exactly one block labeled Refined Prompt: followed by the new prompt.

---

**Example**

Original Prompt:  
A misty forest at dawn, with a lone deer drinking from a stream, soft pastel colors.

Feedback:  
Make the scene feel more mystical—add shafts of light through the trees. Reduce pastel tones and use deeper greens. Emphasize texture on the dee's fur.

Refined Prompt:  
An enchanted forest at first light, deep emerald foliage pierced by golden light shafts, a solitary deer with richly textured fur drinking from a gentle stream, subtle mist curling among mossy stones, painterly realism, high resolution, soft directional lighting.

---

You can swap in any `{original_prompt}` and its `{feedback}` to generate an improved, actionable image-generation prompt every time.
"""


GENERATION_TEXT_PROMPT_REFINED = """You are a multimodal text refinement agent. Your job is to take an existing piece of generated text and improve it by carefully cross-checking against both an audio clip and an image. Do not introduce any details not supported by the audio or image.

[Inputs]
1. `Original Text`
2. Image Information: {image_info}.
3. Audio Information: {audio_info}.


[Task]
1. **Fact-check & Remove Hallucinations**  
   - Identify any statements in the Original Text that are not supported by the audio or image, and remove or correct them.  

2. **Add Missing Details**  
   - If the audio or image contain salient facts or vivid details that the Original Text omits (objects, actions, emotions, ambience), integrate them naturally.  

3. **Ensure Semantic Alignment**  
   - Verify that every claim in the refined text is grounded in the provided modalities.  

4. **Improve Clarity & Flow**  
   - Reorganize sentences for logical progression, smooth transitions, and readability.  

5. **Match Tone & Style**  
   - Align the refined text's tone with the audio's mood and the image's atmosphere (e.g., dramatic, serene, urgent).  

6. **Polish Language**  
   - Correct grammar, tighten phrasing, and eliminate redundancy.

[Output]
Return **only** the fully refined text, as a single coherent passage—no commentary or metadata.  """


GENERATION_AUDIO_PROMPT_REFINED = """
You are a multimodal audio-generation agent. Your job is to take an existing audio generation prompt and refine it by carefully cross-checking against a provided text passage and an image description. Do not introduce any content not supported by the text or image.

[Inputs]  
Original Audio Prompt:  
“{original_audio_prompt}”

Text Information:  
“{text_content}”

Image Information:  
“{image_description}”

[Task]  
1. **Fact-check & Remove Unsupported Content**  
   - Identify any instructions or details in the Original Audio Prompt that are not grounded in the Text Data or Image Data, and remove or correct them.

2. **Integrate Missing Modal Details**  
   - If the text or image contain important details—such as setting, character actions, mood descriptors, or ambient cues—that are missing from the prompt, weave them in naturally.

3. **Specify Sound Source & Format**  
   - Clearly state who or what is producing the sound (e.g., “the sound of [object/action]” for non-speech, or “a man/woman speaks [text]” for dialogue).
   - Indicate desired acoustic environment (e.g., “with soft background rain,” “echoing in a cathedral,” “quiet studio tone”).

4. **Set Tone, Pacing & Emotion**  
   - Based on the mood conveyed by the text and image, prescribe pacing (e.g., slow, measured; quick, urgent), emotional tone (e.g., calm, anxious), and any emphasis or pauses.

5. **Enhance Clarity & Conciseness**  
   - Rewrite for brevity and precision so that the audio generation engine receives a clear, unambiguous instruction.

6. **Polish Language & Structure**  
   - Use vivid, action-oriented language suitable for audio synthesis, and organize details in logical order (source → content → tone → setting).

[Output]  
Return **only** the fully refined audio prompt as a single, self-contained instruction—no commentary or metadata.  
"""