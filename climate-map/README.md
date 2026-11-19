# India Climate Startup Map

An interactive map of 1,818 Indian climate tech startups across 140 cities, 597 of them funded or VC-backed, built from Crustdata, climate VC portfolios and funding news.
Switch between a similarity map, where startups that build similar things cluster together, and an India map that pins each one to its city. Live: https://india-climate-map-beta.vercel.app

- Search by startup, sector, city or founder
- Filter by 16 climate sub-sectors
- Click a dot to see its description, funding, founders, hiring signals and its 10 most similar startups

## How it works

1. Data: ~145 researched startups (VC portfolios such as Rainmatter, Avaana, Sangam, Climate Angels and Omnivore, plus Inc42 and Tracxn) merged with ~1,600 companies from Crustdata semantic search, filtered for climate relevance.
2. Embeddings: each startup's description is turned into a TF-IDF + SVD text vector.
3. Layout: UMAP projects the vectors to 2D, so similar startups land near each other. K-means clusters get auto-generated labels.
4. India map: headquarters geocoded with the GeoNames gazetteer (city, or state when only the state is known), drawn on DataMeet state and district boundaries (CC BY 2.5 India).
5. Front end: a single static HTML page drawing the dots on a canvas. No build step.

Built by [Aditya Bhandare](https://bhandareaditya.github.io/). Data as of November 2026.
