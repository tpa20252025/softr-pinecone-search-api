import express from "express";
import cors from "cors";

const app = express();
app.use(express.json());
app.use(cors({ origin: true }));

const {
  OPENAI_API_KEY,
  PINECONE_API_KEY,
  PINECONE_HOST,
  BUBBLE_SECRET_KEY,
  PINECONE_NAMESPACE = "",
  // SAFETY SWITCH: Set this to "true" only after you finish your Airtable upserts
  ENABLE_HYBRID = "false" 
} = process.env;

// --- HELPERS ---
const trimHost = (u = "") => u.replace(/\/+$/, "");
const DAY_MS = 24 * 60 * 60 * 1000;
const ymd = (d) => new Date(d).toISOString().slice(0, 10);

function ymdRange(fromYMD, toYMD) {
  const out = [];
  let t = new Date(fromYMD + "T00:00:00Z").getTime();
  const end = new Date(toYMD + "T00:00:00Z").getTime();
  for (; t <= end; t += DAY_MS) out.push(ymd(t));
  return out;
}

// (1) DENSE EMBEDDING (Meaning)
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

// (2) SPARSE EMBEDDING (Keywords)
async function getSparseEmbedding(text) {
  // Guard: Never call the API with an empty string
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

    if (!q) return res.status(400).json({ error: "Missing q" });
    const topK = Math.min(parseInt(String(req.query.topK ?? "10"), 10) || 10, 100);

    // STEP 1: Generate Dense Vector
    const denseVector = await getDenseEmbedding(q);

    // STEP 2: Conditional Sparse Generation (The Safety Switch)
    let sparseVector = null;
    // Only attempt sparse embedding if:
    // 1. Expert Hybrid is enabled globally via ENV
    // 2. The user has provided an exact phrase to search for
    if (ENABLE_HYBRID === "true" && exactPhrase) {
      sparseVector = await getSparseEmbedding(exactPhrase);
    }

    // STEP 3: Build Metadata Filter
    const filter = {};
    const rawType = String(req.query.type ?? "").trim().toLowerCase();
    let tokens = rawType.replace(/\//g, ",").split(",").map(s => s.trim()).filter(Boolean);
    
    if (tokens.length && !tokens.includes("all")) {
      const allowedTypeLabels = new Set();
      for (const t of tokens) {
        if (t === "essay" || t === "essays") allowedTypeLabels.add("Essay");
        else if (t === "podcast" || t === "podcasts") {
          allowedTypeLabels.add("Podcast");
          allowedTypeLabels.add("Podcast, Video");
        }
        else if (t === "video" || t === "videos") {
          allowedTypeLabels.add("Video");
          allowedTypeLabels.add("Podcast, Video");
        }
      }
      if (allowedTypeLabels.size) filter.final_type = { $in: Array.from(allowedTypeLabels) };
    }

    const daysNum = parseInt(String(req.query.days ?? ""), 10);
    if (Number.isFinite(daysNum) && daysNum > 0) {
      const allowedDates = ymdRange(ymd(Date.now() - daysNum * DAY_MS), ymd(Date.now()));
      filter.date_of_publication = { $in: allowedDates };
    }

    // STEP 4: Query Pinecone
    // If sparseVector is null, we send a pure dense query to prevent index errors
    const body = {
      vector: sparseVector ? denseVector.map(v => v * alpha) : denseVector,
      topK,
      includeMetadata: true,
      ...(sparseVector && { sparse_vector: sparseVector }), // Include ONLY if safe
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
    const items = (j.matches ?? []).map(m => ({ id: m.id, score: m.score, ...m.metadata }));
    res.json(items);

  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("Expert Hybrid Search API listening on", PORT));
