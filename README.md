# Fashion Recommender

This GitHub repository contains the code and files necessary to run a fashion recommendation system. The system includes several steps to process data, extract image attributes, and deploy a Streamlit app for similarity search and displaying product images.

## One-time Process (Code Not on GitHub)

1. **Install Required Libraries**
    ```sh
    pip install -r requirements.txt
    ```

2. **Create Sample Data**
    - Execute `create_sample.ipynb` to obtain the desired number of products under the specified category by removing skewness.

3. **Scrape Sample Data URLs**
    - Execute `myntra_scraping.ipynb` to scrape the sample data URLs from Myntra.

4. **Extract Image Attributes**
    - Execute `image_attribute_extraction.ipynb` to extract attributes of the main image of the product and store its embeddings.

5. **Convert Attributes to Embeddings and Image to Bytes**
    - Execute `convert_to_embeddings_and_image_bytes.ipynb` to convert the attributes to embeddings and convert the first image of each product to bytes after resizing it into 150x150 pixels.

6. **Chunk Parquet Files**
    - Execute `chunking_parquet_files.ipynb` to chunk the final parquet file into different files with 2700 records each.

7. **Run the Streamlit App**
    - Execute `app.py` which will locate the parquet files to display images on the Streamlit app and use the embeddings for similarity search purposes.

## Repository Links

- **GitHub Repository**: [Fashion Recommender](https://github.com/sanjana3198/FashionRecommender)
- **Streamlit App**: [Midas App](https://midas-app.streamlit.app/)

This GitHub repository contains `app.py` and parquet files stored, which is deployed on Streamlit Cloud Community.

## Disclaimer

We source our data from Myntra.com. Our aim with these fashion recommendations is to guide you towards products based on the following criteria:

- **Product Name**: The specific name of the item.
- **Image Description**:
  - **Color**: Describing the tone, saturation, and brightness of the fabric.
  - **Pattern**: Any designs or motifs present on the fabric, such as stripes, polka dots, floral prints, or geometric shapes.
  - **Detailing**: Any additional decorative elements adorning the garment, such as lace, embroidery, sequins, ruffles, etc.
  - **Neckline**: The shape of the garment's opening around the neck.
  - **Sleeve Length**: Description of the sleeves, indicating if they are long, short, three-quarter length, or sleeveless.
  - **Fit**: How the garment conforms to the body, whether it's loose, tight, baggy, or fitted.
  - **Style**: The overall aesthetic or fashion sense embodied by the garment, encompassing categories like casual, formal, vintage, bohemian, sporty, etc.
  - **Length**: The garment's measurement in terms of how long or short it is, including descriptors like short-sleeved, knee-length, ankle-length, etc.
  - **Functionality**: Special features or practical aspects of the garment, like pockets, zippers, buttons, adjustable straps, etc.
  - **Occasion**: Suitable events or scenarios for wearing the garment, such as work, parties, casual outings, formal events, etc.
  - **Seasonality**: Whether the garment is intended for a particular season, such as lightweight fabrics for summer or heavier materials for winter.
