from annoy import AnnoyIndex
from utils import *

def load_vector_database():
    stored_annoy = AnnoyIndex(VECTOR_SIZE, SIMILARITY)
    stored_annoy.load(DB_NAME)

def search_process(prompt_embeds):
    prompt_embeds = prompt_embeds.flatten()
    return database.get_nns_by_vector(prompt_embeds, SEARCH_SIZE)