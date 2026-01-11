// --- MAIN SEARCH ROUTE ---
app.get("/search", async (req, res) => {
  try {
    // ========== ADD DEBUG CODE HERE (FIRST THING) ==========
    console.log("=== INCOMING REQUEST DEBUG ===");
    console.log("Full URL:", req.url);
    console.log("All query params:", JSON.stringify(req.query, null, 2));
    console.log("type param raw:", req.query.type);
    console.log("==============================");
    // ========== END DEBUG CODE ==========

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

    // 4. Build Metadata Filter (Brute Force Method)
    const filter = {};
    const bubbleType = String(req.query.type ?? "").trim();
    
    // ========== ADD MORE DEBUG HERE ==========
    console.log("=== FILTER LOGIC DEBUG ===");
    console.log("bubbleType after trim:", bubbleType);
    console.log("bubbleType length:", bubbleType.length);
    console.log("bubbleType === 'All Content Types'?", bubbleType === "All Content Types");
    console.log("bubbleType === 'Essays Only'?", bubbleType === "Essays Only");
    console.log("bubbleType === 'Podcasts/Videos Only'?", bubbleType === "Podcasts/Videos Only");
    // ========== END DEBUG CODE ==========

    if (bubbleType === "All Content Types") {
        // (1) Allow anything. No filter added to 'Type'.
        console.log("Matched: All Content Types");
    } 
    else if (bubbleType === "Essays Only") {
        // (2) Allow only specific permutations
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
        // (3) Allow all except those that are "Essays"
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

    // ========== ADD FINAL FILTER DEBUG ==========
    console.log("=== FINAL FILTER ===");
    console.log("Filter object:", JSON.stringify(filter, null, 2));
    console.log("Filter has keys?", Object.keys(filter).length > 0);
    console.log("====================");
    // ========== END DEBUG CODE ==========

    // 5. Query Pinecone
    const body = {
      vector: sparseVector ? denseVector.map(v => v * alpha) : denseVector,
      topK,
      includeMetadata: true,
      ...(sparseVector && { sparse_vector: sparseVector }),
      ...(PINECONE_NAMESPACE && { namespace: PINECONE_NAMESPACE }),
      ...(Object.keys(filter).length ? { filter } : {})
    };

    // ========== ADD BODY DEBUG ==========
    console.log("=== PINECONE REQUEST ===");
    console.log("Body being sent (without vector):", JSON.stringify({
      ...body,
      vector: `[array of ${body.vector.length} numbers]`
    }, null, 2));
    console.log("========================");
    // ========== END DEBUG CODE ==========

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
    
    // ========== ADD RESULTS DEBUG ==========
    console.log("=== RESULTS DEBUG ===");
    console.log("Number of results:", items.length);
    if (items.length > 0) {
      console.log("First result Type:", items[0].type);
      console.log("All unique Types in results:", [...new Set(items.map(i => i.type))]);
    }
    console.log("=====================");
    // ========== END DEBUG CODE ==========
    
    res.json(items);

  } catch (e) {
    console.error("=== ERROR ===", e);
    res.status(500).json({ error: String(e) });
  }
});
```

## After Adding This Code

1. **Commit and push to GitHub**
2. **Render will auto-deploy** (or manually deploy)
3. **Go to Render dashboard → your service → Logs**
4. **Make a search from Bubble** with "Essays Only" selected
5. **Watch the logs** - you'll see all the debug output

This will tell you:
- What exact value Bubble is sending for `type`
- Whether the if conditions are matching
- What filter is being sent to Pinecone
- What Type values are coming back in results

The logs will look something like:
```
=== INCOMING REQUEST DEBUG ===
Full URL: /search?q=education&type=Essays%20Only&topK=10
All query params: {
  "q": "education",
  "type": "Essays Only",
  "topK": "10"
}
type param raw: Essays Only
==============================
=== FILTER LOGIC DEBUG ===
bubbleType after trim: Essays Only
bubbleType length: 11
bubbleType === 'Essays Only'? true
Matched: Essays Only
=== FINAL FILTER ===
Filter object: {
  "Type": {
    "$in": ["Essay", "Essay, Podcast", ...]
  }
}
