import express from "express";
import cors from "cors";

const app = express();
app.use(express.json());
// tighten this to your Bubble/Softr origin later
app.use(cors({ origin: true }));

const {
  OPENAI_API_KEY,
  PINECONE_API_KEY,
  PINECONE_HOST,            // e.g. https://<index>-xxxx.svc.us-east-1-aws.pinecone.io
  PINECONE_NAMESPACE = ""
} = process.env;

const MODEL = "text-embedding-3-small"; // 1536-dim cosine
const DAY_MS = 24 * 60 * 60 * 1000;

const trimHost = (u = "") => u.replace(/\/+$/, "");

async function embed(text) {
  const r = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model: MODEL, input: text }),
  });
  const j = await r.json();
  if (!r.ok) throw new Error(`OpenAI: ${r.status} ${JSON.stringify(j)}`);
  return j.data[0].embedding;
}

app.get("/healthz", (_req, res) => res.json({ ok: true }));

// GET /search?q=...&topK=30&type=essay|podcast|video|podcast,video&days=30
app.get("/search", async (req, res) => {
  try {
    const q = String(req.query.q ?? "").trim();
    if (!q) return res.status(400).json({ error: "Missing q" });

    const topK =
      Math.min(parseInt(String(req.query.topK ?? "10"), 10) || 10, 50);

    // ---- TYPE PARSING ----
    // Accept "essay", "podcast", "video" or CSV like "podcast,video"
    const rawType = String(req.query.type ?? "").trim().toLowerCase();
    let tokens = rawType
      .replace(/\//g, ",")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);

    if (tokens.includes("all")) tokens = [];

    // Build allowed labels for your stored values
    // final_type ∈ {"Essay","Podcast","Video","Podcast, Video"}
    const allowedTypeLabels = new Set();
    for (const t of tokens) {
      if (t === "essay" || t === "essays") {
        allowedTypeLabels.add("Essay");
      } else if (t === "podcast" || t === "podcasts") {
        allowedTypeLabels.add("Podcast");
        allowedTypeLabels.add("Podcast, Video"); // include combo
      } else if (t === "video" || t === "videos") {
        allowedTypeLabels.add("Video");
        allowedTypeLabels.add("Podcast, Video"); // include combo
      }
    }
    const finalTypeValues =
      allowedTypeLabels.size ? Array.from(allowedTypeLabels) : null;

    // ---- DAYS PARSING ----
    const daysNum = parseInt(String(req.query.days ?? ""), 10);
    const days =
      Number.isFinite(daysNum) && daysNum > 0 ? daysNum : null;
    const cutoffYMD = days
      ? new Date(Date.now() - days * DAY_MS).toISOString().slice(0, 10)
      : null;

    // ---- EMBED & QUERY ----
    const vector = await embed(q);

    const filter = {};
    if (finalTypeValues) filter.final_type = { $in: finalTypeValues };
    if (cutoffYMD)
      filter.date_of_publication = { $gte: cutoffYMD };

    const body = {
      vector,
      topK,
      includeMetadata: true,
      ...(PINECONE_NAMESPACE && { namespace: PINECONE_NAMESPACE }),
      ...(Object.keys(filter).length ? { filter } : {}),
    };

    const r = await fetch(`${trimHost(PINECONE_HOST)}/query`, {
      method: "POST",
      headers: {
        "Api-Key": PINECONE_API_KEY,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
    const j = await r.json();
    if (!r.ok) {
      return res.status(500).json({ error: "Pinecone", detail: j });
    }

    const items = (j.matches ?? []).map((m) => ({
      id: m.id,
      score: m.score,
      ...m.metadata,
    }));

    res.json(items);
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log("search api listening on", PORT);
});
