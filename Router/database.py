from main import mongo_client

def create_user_collections(mongo_client, user_id: str, role: str):
    db = mongo_client['user_accounts']
    if not mongo_client:
        raise RuntimeError("MongoDB client database 未正確初始化")
    
    if role == "Farmer":
        collections = ["farm_info", "growth_prediction"]
    elif role == "GeneticResearcher":
        collections = ["personal_info", "gene_prediction"]
    else:
        raise ValueError("Invalid role")

    for name in collections:
        if name not in db.list_collection_names():
            db.create_collection(name)