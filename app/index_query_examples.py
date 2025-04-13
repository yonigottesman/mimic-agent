import json
from pathlib import Path

import lancedb
from lancedb.pydantic import LanceModel, Vector
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


class ExampleQuerySchema(LanceModel):
    concept: str
    query_description: str
    query_sql: str
    query_description_vector: Vector(768)


def main():
    model = SentenceTransformer("Alibaba-NLP/gte-modernbert-base")

    data = []
    concepts = json.load(open("app/resources/concepts/concepts.json"))

    for concept in tqdm(concepts["concepts"]):
        vector = model.encode(concept["description"])
        query_text = (Path("app/resources/concepts") / concept["link"]).read_text()
        data.append(
            ExampleQuerySchema(
                concept=concept["link"],
                query_description=concept["description"],
                query_sql=query_text,
                query_description_vector=vector,
            )
        )

    uri = "app/resources/lancedb"
    db = lancedb.connect(uri)
    table = db.create_table("example_queries", data=data, schema=ExampleQuerySchema)
    table.create_fts_index("query_description", use_tantivy=True, replace=True)


if __name__ == "__main__":
    main()
