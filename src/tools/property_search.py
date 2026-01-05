import os
import pandas as pd
import superlinked.framework as sl
from langchain_core.tools import tool


class Property(sl.Schema):
    id : sl.IdField
    description : sl.String
    baths : sl.Float
    rooms : sl.Integer
    sqft : sl.Float 
    location : sl.String
    price : sl.Float    

# Now create the instance
property_schema = Property()

# --- SEARCH ENGINE SETUP ---
description_space = sl.TextSimilaritySpace(
    text=property_schema.description, 
    model="sentence-transformers/all-MiniLM-L6-v2"
)
price_space = sl.NumberSpace(
    number=property_schema.price, 
    min_value=50000, 
    max_value=20000000, 
    mode=sl.Mode.MINIMUM
)

property_index = sl.Index(spaces=[description_space, price_space])

superlinked_app = None

def init_superlinked():
    global superlinked_app
    
    openai_config = sl.OpenAIClientConfig(
    api_key=os.getenv("GROQ_API_KEY"), 
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1"
)
    try:
        # Python 3.14 optimized executor
        source = sl.InMemorySource(property_schema, parser=sl.DataFrameParser(schema=property_schema))
        executor = sl.InMemoryExecutor(sources=[source], indices=[property_index])
        superlinked_app = executor.run()

        # Dynamic Pathing for your project structure
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.normpath(os.path.join(current_dir, "..", "..", "data", "properties.csv"))

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['id'] = df['id'].astype(str)
            source.put([df])
            print("✅ SUCCESS: Sarah's 3.14 engine is live.")
        else:
            print(f"❌ ERROR: CSV not found at {csv_path}")
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR in init_superlinked: {e}")
        import traceback
        traceback.print_exc()

@tool
def search_properties(user_request: str):
    """Search for real estate properties using natural language."""
    global superlinked_app
    if superlinked_app is None:
        init_superlinked()
        
    query = (
        sl.Query(property_index)
        .find(property_schema)
        .similar(description_space, sl.Param("n_query"))
        .limit(1)
    )
    
    try:
        results = superlinked_app.query(query, n_query=user_request)
        pdf = sl.PandasConverter.to_pandas(results)
        # Write columns to a file to be sure
        with open("columns.txt", "w") as f:
            f.write(str(list(pdf.columns)))

        print(f"DEBUG: Dataframe columns: {list(pdf.columns)}")
        
        if pdf.empty:
            return "I'm sorry, I couldn't find any properties matching that."
        
        res = pdf.iloc[0]
        return f"I found a property in {match_details['location']} for {match_details['price']} euros. It has {match_details['rooms']} rooms."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"An error occurred while searching: {e}"