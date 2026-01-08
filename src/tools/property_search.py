import os
import pandas as pd
import superlinked.framework as sl
from langchain_core.tools import tool
from pathlib import Path
import logging
import inflect

# --- INITIALIZE LOGGING & UTILS ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Sarah-Search")
p = inflect.engine()

def format_price_for_tts(amount):
    """Converts numeric price to spoken words for natural TTS output."""
    try:
        if amount == 'Unknown': return amount
        num_words = p.number_to_words(int(amount))
        return f"{num_words} euros"
    except:
        return f"{amount} euros"

# --- SCHEMA & INDEX DEFINITION (UNCHANGED) ---
class Property(sl.Schema):
    id : sl.IdField
    description : sl.String
    baths : sl.Float
    rooms : sl.Integer
    sqft : sl.Float
    location : sl.String
    price : sl.Float

property_schema = Property()
description_space = sl.TextSimilaritySpace(text=property_schema.description, model="sentence-transformers/all-MiniLM-L6-v2")
price_space = sl.NumberSpace(number=property_schema.price, min_value=50000, max_value=20000000, mode=sl.Mode.MINIMUM)
property_index = sl.Index(spaces=[description_space, price_space])

superlinked_app = None

def init_superlinked():
    global superlinked_app
    print("🚀 Pre-heating Sarah's search engine...")
    try:
        source = sl.InMemorySource(property_schema, parser=sl.DataFrameParser(schema=property_schema))
        executor = sl.InMemoryExecutor(sources=[source], indices=[property_index])
        superlinked_app = executor.run()
        root_dir = Path(__file__).resolve().parents[2]
        csv_path = root_dir / "data" / "properties.csv"

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['id'] = df['id'].astype(str)
            source.put([df])
            print("✅ SUCCESS: Sarah's 3.14 engine is live.")
        else:
            print(f"❌ ERROR: CSV not found at {csv_path}")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")

@tool
def search_properties(user_request: str):
    """
    Search for real estate properties using natural language. 
    Returns details about the best matching property.
    """
    global superlinked_app
    
    # 1. TERMINAL VISIBILITY
    print(f"\n🔍 [SARAH DEBUG]: Activating Superlinked for query: '{user_request}'")
    logger.info(f"🚀 [SUPERLINKED ACTIVATED]: Processing semantic search...")

    if superlinked_app is None:
        init_superlinked()
        
    query = (
        sl.Query(property_index)
        .find(property_schema)
        .similar(description_space, sl.Param("n_query"))
        .select_all()
        .limit(1)
    )
    
    try:
        results = superlinked_app.query(query, n_query=user_request)
        pdf = sl.PandasConverter.to_pandas(results)
        
        if pdf.empty:
            return "No properties found matching the criteria."
        
        res = pdf.iloc[0]
        
        # 2. PRICE NORMALIZATION FOR TTS
        raw_price = res.get('price', 'Unknown')
        spoken_price = format_price_for_tts(raw_price)
        
        property_data = {
            "status": "Search Successful",
            "location": res.get('location', 'Unknown'),
            "price_numeric": raw_price,
            "price_spoken": spoken_price, # Sarah will now use this for speech
            "rooms": res.get('rooms', 'Unknown'),
            "description": res.get('description', 'No description available'),
            "id": res.get('id', 'Unknown')
        }
        
        logger.info(f"✅ [SUPERLINKED]: Found match in {property_data['location']}")
        return f"Property Metadata: {property_data}"

    except Exception as e:              
        return f"I encountered a technical glitch while searching: {e}"