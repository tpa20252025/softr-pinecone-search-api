import express from "express";
import cors from "cors";
import fetch from "node-fetch";

const app = express();
app.use(express.json());
app.use(cors({ origin: true }));

const {
  OPENAI_API_KEY,
  PINECONE_API_KEY,
  PINECONE_HOST,
  BUBBLE_SECRET_KEY,
  PINECONE_NAMESPACE = "",
  ENABLE_HYBRID = "false" 
} = process.env;

// --- HELPERS ---
const trimHost = (u = "") => u.replace(/\/+$/, "");
const DAY_MS = 24 * 60 * 60 * 1000;

// REPLACED: Helper to generate 4 variations of the date to catch "5/1/2024" AND "05/01/2024"
const getMultiFormats = (dateObj) => {
  const y = dateObj.getUTCFullYear();
  const m = dateObj.getUTCMonth() + 1; // Raw number (1-12)
  const d = dateObj.getUTCDate();      // Raw number (1-31)
  
  const mm = String(m).padStart(2, '0'); // Padded (01-12)
  const dd = String(d).padStart(2, '0'); // Padded (01-31)
  
  return [
    `${y}-${mm}-${dd}`, // ISO Strict: 2024-05-01
    `${mm}/${dd}/${y}`, // US Padded:  05/01/2024
    `${m}/${d}/${y}`,   // US Loose:   5/1/2024  <-- This is likely the missing one
    `${m}/${dd}/${y}`,  // Mixed 1:    5/01/2024 (Just in case)
    `${mm}/${d}/${y}`   // Mixed 2:    05/1/2024 (Just in case)
  ];
};

function getMultiDateRange(daysBack) {
  const out = new Set(); // Use Set to avoid duplicate strings
  const now = new Date();
  
  // Normalize to start of today in UTC
  const endTimestamp = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
  let currentTimestamp = endTimestamp - (daysBack * DAY_MS);

  while (currentTimestamp <= endTimestamp) {
    const d = new Date(currentTimestamp);
    // Get all 5 variations for this single day
    const variations = getMultiFormats(d);
    variations.forEach(v => out.add(v));
    
    currentTimestamp += DAY_MS;
  }
  return Array.from(out);
}

// (1) DENSE EMBEDDING
async function getDenseEmbedding(text) {
  const r = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: { Authorization: `Bearer ${OPENAI_API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({ model: "text-embedding-3-small", input: text }),
  });
  const j = await r.json();
  if (!r.ok) throw new Error(`OpenAI Error: ${r.status}`);
  return j.data[0].embedding;
}

// (2) SPARSE EMBEDDING
async function getSparseEmbedding(text) {
  if (!text || text.trim() === "") return null;
  const r = await fetch("https://api.pinecone.io/embed", {
    method: "POST",
    headers: { 
      "Api-Key": PINECONE_API_KEY, 
      "Content-Type": "application/json",
      "X-Pinecone-Api-Version": "2024-10" 
    },
    body: JSON.stringify({
      model: "pinecone-sparse-english-v0",
      parameters: { input_type: "query" },
      inputs: [{ text }]
    })
  });
  const j = await r.json();
  if (!r.ok) throw new Error(`Pinecone Sparse Error: ${r.status}`);
  return j.data[0].sparse_values;
}

// --- SECURITY MIDDLEWARE ---
const checkAuth = (req, res, next) => {
  const authHeader = req.headers['authorization']; 
  if (!BUBBLE_SECRET_KEY) return res.status(500).json({ error: 'Server config error' });
  if (!authHeader || authHeader !== `Bearer ${BUBBLE_SECRET_KEY}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  next(); 
};

app.use(checkAuth);
app.get("/healthz", (_req, res) => res.json({ ok: true }));

// --- MAIN SEARCH ROUTE ---
app.get("/search", async (req, res) => {
  try {
    const q = String(req.query.q ?? "").trim();
    const alpha = parseFloat(req.query.alpha ?? "0.8");
    const exactPhrase = String(req.query.exact ?? "").trim();

    // 1. Validate Input
    if (!q) return res.status(400).json({ error: "Missing q" });
    const topK = Math.min(parseInt(String(req.query.topK ?? "10"), 10) || 10, 100);

    // 2. Generate Dense Vector
    const denseVector = await getDenseEmbedding(q);

    // 3. Generate Sparse Vector (Optional)
    let sparseVector = null;
    if (ENABLE_HYBRID === "true" && exactPhrase) {
      sparseVector = await getSparseEmbedding(exactPhrase);
    }

    // 4. Build Metadata Filter
    const filter = {};
    const bubbleType = String(req.query.type ?? "").trim(); 

    if (bubbleType === "All Content Types") {
        // (1) Allow anything.
    } 
    else if (bubbleType === "Essays Only") {
        filter.Type = { 
            $in: [
                "Essay", "Essay, Podcast", "Essay, Video", 
                "Essay, Podcast, Video", "Essay, Video, Podcast", 
                "Podcast, Essay", "Podcast, Essay, Video", 
                "Podcast, Video, Essay", "Video, Essay", 
                "Video, Essay, Podcast", "Video, Podcast, Essay"
            ] 
        };
    } 
    else if (bubbleType === "Podcasts/Videos Only") {
        filter.Type = { $nin: ["Essay", "Essays"] };
    }

    // handle Date filtering (UPDATED: MULTI-FORMAT SUPPORT)
    const daysNum = parseInt(String(req.query.days ?? ""), 10);
    if (Number.isFinite(daysNum) && daysNum > 0) {
      // Use the new helper that catches 5/1/2024 and 05/01/2024
      const allowedDates = getMultiDateRange(daysNum);
      filter["Date of Publication"] = { $in: allowedDates };
    }

    // 5. Query Pinecone
    const body = {
      vector: sparseVector ? denseVector.map(v => v * alpha) : denseVector,
      topK,
      includeMetadata: true,
      ...(sparseVector && { sparse_vector: sparseVector }),
      ...(PINECONE_NAMESPACE && { namespace: PINECONE_NAMESPACE }),
      ...(Object.keys(filter).length ? { filter } : {})
    };

    const r = await fetch(`${trimHost(PINECONE_HOST)}/query`, {
      method: "POST",
      headers: { "Api-Key": PINECONE_API_KEY, "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });

    const j = await r.json();
    if (!r.ok) return res.status(500).json({ error: "Pinecone Error", detail: j });

    // 6. Map Response
    const items = (j.matches ?? []).map(m => ({ 
      id: m.id, 
      score: m.score, 
      
      title:       m.metadata.Title,
      author:      m.metadata.Author,
      url:         m.metadata.URL,
      publication: m.metadata.Publication,
      type:        m.metadata.Type,
      date:        m.metadata["Date of Publication"],
      snippet:     m.metadata.Snippet,
      ai_summary:  m.metadata["AI Summary"],
      abstract:    m.metadata["AI Summary"] || m.metadata.Snippet,

      ...m.metadata 
    }));
    
    res.json(items);

  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("Expert Hybrid Search API listening on", PORT));
