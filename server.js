import express from "express";
import cors from "cors";

const app = express();
app.use(express.json());
// tighten this later to your Softr domain(s)
app.use(cors({ origin: true }));

const {
  OPENAI_API_KEY,
  PINECONE_API_KEY,
  PINECONE_HOST,            // e.g. https://<index>-xxxx.svc.us-east-1-aws.pinecone.io
  PINECONE_NAMESPACE = ""   // optional
} = process.env;

const MODEL = "text-embedding-3-small"; // 1536-dim cosine

function trimHost(u){ return (u || "").replace(/\/+$/,""); }

async function embed(text){
  const r = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: { "Authorization": `Bearer ${OPENAI_API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({ model: MODEL, input: text })
  });
  const j = await r.json();
  if (!r.ok) throw new Error(`OpenAI: ${r.status} ${JSON.stringify(j)}`);
  return j.data[0].embedding;
}

app.get("/healthz", (_req,res) => res.json({ ok:true }));

// GET /search?q=your+query&topK=10
app.get("/search", async (req,res) => {
  try {
    const q = (req.query.q || "").toString().trim();
    const topK = Math.min(parseInt(req.query.topK || "10", 10), 50);
    if (!q) return res.status(400).json({ error: "Missing q" });

    const vector = await embed(q);

    const body = {
      vector,
      topK,
      includeMetadata: true,
      ...(PINECONE_NAMESPACE && { namespace: PINECONE_NAMESPACE })
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
      ...m.metadata     // <- your keys: final_title, final_author, etc.
    }));

    res.json(items);
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("search api listening on", PORT));
