import express from "express";
import cors from "cors";

const app = express();
app.use(express.json());
app.use(cors({ origin: true })); // tighten to your Bubble/Softr origin later

const {
  OPENAI_API_KEY,
  PINECONE_API_KEY,
  PINECONE_HOST,            // e.g. https://<index>-xxxx.svc.us-east-1-aws.pinecone.io
  PINECONE_NAMESPACE = ""
} = process.env;
// --- SECURITY CHECK START ---
const checkAuth = (req, res, next) => {
  const secretKey = process.env.BUBBLE_SECRET_KEY; // Must match the Key Name you used in Render
  const authHeader = req.headers['authorization']; 

  // If the secret key isn't set in Render yet, log a warning but don't crash
  if (!secretKey) {
    console.error("WARNING: BUBBLE_SECRET_KEY is missing in Render environment variables!");
    return res.status(500).json({ error: 'Server configuration error' });
  }

  // Check if the header matches "Bearer <YOUR_SECRET>"
  if (!authHeader || authHeader !== `Bearer ${secretKey}`) {
    return res.status(401).json({ error: 'Unauthorized: Missing or wrong password' });
  }
  next(); 
};

// Apply the lock to ALL routes
app.use(checkAuth);
// --- SECURITY CHECK END ---
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

// make an array of YYYY-MM-DD strings from start to end inclusive (UTC)
function ymd(d) { return new Date(d).toISOString().slice(0, 10); }
function ymdRange(fromYMD, toYMD) {
  const out = [];
  let t = new Date(fromYMD + "T00:00:00Z").getTime();
  const end = new Date(toYMD + "T00:00:00Z").getTime();
  for (; t <= end; t += DAY_MS) out.push(ymd(t));
  return out;
}

app.get("/healthz", (_req, res) => res.json({ ok: true }));

// GET /search?q=...&topK=30&type=essay|podcast|video|podcast,video&days=30
app.get("/search", async (req, res) => {
  try {
    const q = String(req.query.q ?? "").trim();
    if (!q) return res.status(400).json({ error: "Missing q" });
    const topK = Math.min(parseInt(String(req.query.topK ?? "10"), 10) || 10, 100);

    // ---- TYPE PARSING ----
    // Accept "essay", "podcast", "video" or CSV like "podcast,video"
    const rawType = String(req.query.type ?? "").trim().toLowerCase();
    let tokens = rawType
      .replace(/\//g, ",")
      .split(",")
      .map(s => s.trim())
      .filter(Boolean);
    if (tokens.includes("all")) tokens = [];

    // Your stored labels are EXACTLY: "Essay", "Podcast", "Video", "Podcast, Video"
    const allowedTypeLabels = new Set();
    for (const t of tokens) {
      if (t === "essay" || t === "essays")       allowedTypeLabels.add("Essay");
      else if (t === "podcast" || t === "podcasts") {
        allowedTypeLabels.add("Podcast");
        allowedTypeLabels.add("Podcast, Video"); // combo includes podcasts
      }
      else if (t === "video" || t === "videos") {
        allowedTypeLabels.add("Video");
        allowedTypeLabels.add("Podcast, Video"); // combo includes videos
      }
    }
    const finalTypeValues = allowedTypeLabels.size ? Array.from(allowedTypeLabels) : null;

    // ---- DAYS PARSING ---- (string metadata requires $in, not $gte)
    const daysNum = parseInt(String(req.query.days ?? ""), 10);
    const days = Number.isFinite(daysNum) && daysNum > 0 ? daysNum : null;

    const vector = await embed(q);

    // ---- BUILD FILTER ----
    const filter = {};
    if (finalTypeValues) {
      filter.final_type = { $in: finalTypeValues };
    }
    if (days) {
      const fromYMD = ymd(Date.now() - days * DAY_MS);
      const toYMD   = ymd(Date.now());
      const allowedDates = ymdRange(fromYMD, toYMD); // ["2025-08-20", "2025-08-21", ...]
      filter.date_of_publication = { $in: allowedDates };
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

    const items = (j.matches ?? []).map(m => ({
      id: m.id,
      score: m.score,
      ...m.metadata
    }));

    res.json(items);
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("search api listening on", PORT));

