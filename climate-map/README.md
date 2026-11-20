# India Climate Startup Map

A searchable map of 1,818 Indian climate tech startups across 16 sectors and 140 cities, 597 of them funded.
Three views: one clusters startups that do similar work, one pins each to its city on a map of India, and a list you can sort and export to CSV. Live: https://india-climate-map-beta.vercel.app

- Search by startup, sector, city, state or founder (press `/` to start)
- Filter by stage, sector, city, founding year, team size and hiring signal, with live counts on every option
- Quick starts for common questions: funded startups, Bengaluru, EVs and batteries, carbon data and more
- List view that sorts by any column and exports whatever you've filtered as a CSV
- Shareable links: the view, filters and selected startup all live in the URL
- Click any startup for its funding, founders, hiring signals, links and its 10 most similar startups
- Suggest a startup or send a correction straight from the site

## How it works

1. Data: ~145 researched startups (VC portfolios such as Rainmatter, Avaana, Sangam, Climate Angels and Omnivore, plus Inc42 and Tracxn) merged with ~1,600 companies from Crustdata semantic search, filtered for climate relevance.
2. Embeddings: each startup's description is turned into a TF-IDF + SVD text vector.
3. Layout: UMAP projects the vectors to 2D, so similar startups land near each other. K-means clusters get auto-generated labels.
4. India map: headquarters geocoded with the GeoNames gazetteer (city, or state when only the state is known), drawn on DataMeet state and district boundaries (CC BY 2.5 India).
5. Front end: a single static HTML page drawing the dots on a canvas. No build step.

Built by [Aditya Bhandare](https://bhandareaditya.github.io/). Data as of November 2026.
