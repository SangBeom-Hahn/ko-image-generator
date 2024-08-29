from annoy import AnnoyIndex
from utils import *

def load_vector_database():
    stored_annoy = AnnoyIndex(VECTOR_SIZE, SIMILARITY)
    stored_annoy.load(DB_NAME)
    return stored_annoy

def search_process(prompt_embeds):
    database = load_vector_database()
    prompt_embeds = prompt_embeds.flatten()
    return database.get_nns_by_vector(prompt_embeds, SEARCH_SIZE)

def create_vector_database():
    database = AnnoyIndex(VECTOR_SIZE, SIMILARITY)
    pipeline = get_pipeline(get_clip_model(), get_clip_processor())

    for idx, prompt_description in enumerate(prompt_descriptions):
        generator = torch.manual_seed(SEEDS[idx])

        (prompt_embeds, _) = pipeline.encode_prompt(generate_prompt, "cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)
        prompt_embeds = prompt_embeds.flatten()
        annoy_index.add_item(idx, prompt_embeds)

    annoy_index.build(10)
    annoy_index.save('hanbok.annoy')