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
const ymd = (d) => new Date(d).toISOString().slice(0, 10);

function ymdRange(fromYMD, toYMD) {
  const out = [];
  let t = new Date(fromYMD + "T00:00:00Z").getTime();
  const end = new Date(toYMD + "T00:00:00Z").getTime();
  for (; t <= end; t += DAY_MS) out.push(ymd(t));
  return out;
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
    console.log("=== INCOMING REQUEST DEBUG ===");
    console.log("Full URL:", req.url);
    console.log("All query params:", JSON.stringify(req.query, null, 2));
    console.log("type param raw:", req.query.type);
    console.log("==============================");

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
    
    console.log("=== FILTER LOGIC DEBUG ===");
    console.log("bubbleType after trim:", bubbleType);
    console.log("bubbleType length:", bubbleType.length);
    console.log("bubbleType === 'All Content Types'?", bubbleType === "All Content Types");
    console.log("bubbleType === 'Essays Only'?", bubbleType === "Essays Only");
    console.log("bubbleType === 'Podcasts/Videos Only'?", bubbleType === "Podcasts/Videos Only");

    if (bubbleType === "All Content Types") {
        console.log("Matched: All Content Types");
    } 
    else if (bubbleType === "Essays Only") {
        console.log("Matched: Essays Only");
        filter.Type = { 
            $in: [
                "Essay", 
                "Essay, Podcast", 
                "Essay, Video", 
                "Essay, Podcast, Video", 
                "Essay, Video, Podcast", 
                "Podcast, Essay", 
                "Podcast, Essay, Video", 
                "Podcast, Video, Essay", 
                "Video, Essay", 
                "Video, Essay, Podcast", 
                "Video, Podcast, Essay"
            ] 
        };
    } 
    else if (bubbleType === "Podcasts/Videos Only") {
        console.log("Matched: Podcasts/Videos Only");
        filter.Type = { $nin: ["Essay", "Essays"] };
    } else {
        console.log("NO MATCH - bubbleType didn't match any condition");
    }

    // handle Date filtering (independent of Type)
    const daysNum = parseInt(String(req.query.days ?? ""), 10);
    if (Number.isFinite(daysNum) && daysNum > 0) {
      const allowedDates = ymdRange(ymd(Date.now() - daysNum * DAY_MS), ymd(Date.now()));
      filter["Date of Publication"] = { $in: allowedDates };
    }

    console.log("=== FINAL FILTER ===");
    console.log("Filter object:", JSON.stringify(filter, null, 2));
    console.log("Filter has keys?", Object.keys(filter).length > 0);
    console.log("====================");

    // 5. Query Pinecone
    const body = {
      vector: sparseVector ? denseVector.map(v => v * alpha) : denseVector,
      topK,
      includeMetadata: true,
      ...(sparseVector && { sparse_vector: sparseVector }),
      ...(PINECONE_NAMESPACE && { namespace: PINECONE_NAMESPACE }),
      ...(Object.keys(filter).length ? { filter } : {})
    };

    console.log("=== PINECONE REQUEST ===");
    console.log("Sending filter to Pinecone:", body.filter ? JSON.stringify(body.filter, null, 2) : "NO FILTER");
    console.log("========================");

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
    
    console.log("=== RESULTS DEBUG ===");
    console.log("Number of results:", items.length);
    if (items.length > 0) {
      console.log("First result Type:", items[0].type);
      console.log("All unique Types in results:", [...new Set(items.map(i => i.type))]);
    }
    console.log("=====================");
    
    res.json(items);

  } catch (e) {
    console.error("=== ERROR ===", e);
    res.status(500).json({ error: String(e) });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("Expert Hybrid Search API listening on", PORT));
