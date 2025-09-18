import express from "express";
import cors from "cors";

const app = express();
app.use(express.json());
// tighten to your Bubble/Softr origins later
app.use(cors({ origin: true }));

const {
  OPENAI_API_KEY,
  PINECONE_API_KEY,
  PINECONE_HOST,            // e.g. https://<index>-xxxx.svc.us-east-1-aws.pinecone.io
  PINECONE_NAMESPACE = ""
} = process.env;

const MODEL = "text-embedding-3-small"; // 1536-dim cosine

function trimHost(u){ return (u || "").replace(/\/+$/,""); }

async function embed(text){
  const r = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ model: MODEL, input: text })
  });
  const j = await r.json();
  if (!r.ok) throw new Error(`OpenAI: ${r.status} ${JSON.stringify(j)}`);
  return j.data[0].embedding;
}

app.get("/healthz", (_req,res) => res.json({ ok:true }));

// GET /search?q=...&topK=30&type=essay|podcast|video|podcast,video&days=30
app.get("/search", async (req,res) => {
  try {
    const q = (req.query.q || "").toString().trim();
    const topK = Math.min(parseInt(req.query.topK || "10", 10), 50);
    if (!q) return res.status(400).json({ error: "Missing q" });

    // ---- type parsing ----
    // Accept "essay", "podcast", "video", or CSV like "podcast,video".
    const rawType = (req.query.type || "").toString().trim();
    let typeTokens = rawType
      .replace(/\//g, ",")
      .split(",")
      .map(s => s.trim().toLowerCase())
      .filter(Boolean);

    // 'all' disables type filter
    if (typeTokens.includes("all")) typeTokens = [];

    // Build allowed labels for Pinecone's final_type
    // (exact stored labels: "Essay", "Podcast", "Video", "Podcast, Video")
    const allowedTypeLabels = new Set();
    for (const t of typeTokens) {
      if (t === "essay") {
        allowedTypeLabels.add("Essay");
      } else if (t === "podcast") {
        allowedTypeLabels.add("Podcast");
        allowedTypeLabels.add("Podcast, Video");
      } else if (t === "video") {
        allowedTypeLabels.add("Video");
        allowedTypeLabels.add("Podcast, Video");
      }
    }
    const finalTypeFilterValues =
      allowedTypeLabels.size ? Array.from(allowedTypeLabels) : null;

    // ---- days parsing ----
    const daysParam = parseInt((req.query.days || "").toString(), 10);
    const days = Number.isFinite(daysParam) && daysParam > 0 ? daysParam : null;
    const cutoffYMD = days
      ? new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString().slice(0, 10)
      : null;

    const vector = await embed(q);

    // ---- build Pinecone filter ----
    const filter = {};
    if (finalTypeFilterValues) {
      filter["final_type"] = { $in: finalTypeFilterValues };
    }
    if (cutoffYMD) {
      // your field is date_of_publication in YYYY-MM-DD
      filter["date_of_publication"] = { $gte: cutoffYMD };
    }

    const body = {
      vector,
      topK,
      includeMetadata: true,
      ...(PINECONE_NAMESPACE && { namespace: PINECONE_NAMESPACE }),
      ...(Object.keys(filter).length ? { filter } : {})
    };

    const r = await fetch(`${trimHost(PINECONE_HOST)}/query`, {
      method: "POST",
      headers: { "Api-Key": PINECONE_API_KEY, "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    const j = await r.json();
    if (!r.ok) return res.status(500).json({ error: "Pinecone", detail: j });

    const items = (j.matches || []).map(m => ({
      id: m.id,
      score: m.score,
      ...m.metadata // final_title, final_subtitle, final_author, final_publication, final_essay_url, final_image_url, final_type, date_of_publication, etc.
    }));

    res.json(items);
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("search api listening on", PORT));
);
