/* Transparent local fallback; this is deliberately a lexicon baseline, not LSTM/GRU inference. */
(() => {
  const review = document.getElementById("review");
  const counter = document.getElementById("counter");
  const result = document.getElementById("demo-result");
  const maxLength = 2000;
  const example = "The performances are patient and precise, and the ending earns its warmth without becoming sentimental.";

  const positiveWords = new Set([
    "amazing", "appealing", "astonishing", "attractive", "beautiful", "best", "brilliant",
    "captivating", "charming", "clever", "compelling", "delight", "delightful", "dynamic",
    "effective", "enjoyable", "entertaining", "excellent", "exceptional", "exciting",
    "expert", "extraordinary", "fantastic", "fascinating", "favorite", "favourite", "fine",
    "flawless", "fun", "funny", "gem", "good", "gorgeous", "graceful", "great", "gripping",
    "heartfelt", "hilarious", "honest", "impressive", "incredible", "innovative", "insightful",
    "inspiring", "intelligent", "inventive", "like", "liked", "likes", "liking", "lively",
    "love", "loved", "lovely", "loves", "loving", "luminous", "magnificent", "masterful",
    "masterpiece", "memorable", "moving", "original", "outstanding", "passion", "passionate",
    "perfect", "perfection", "phenomenal", "pleased", "pleasing", "pleasure", "polished",
    "poignant", "positive", "powerful", "praise", "radiant", "recommend", "recommended",
    "refreshing", "remarkable", "resonant", "rich", "riveting", "satisfying", "sharp",
    "smart", "solid", "spectacular", "splendid", "stellar", "striking", "strong", "stunning",
    "sublime", "subtle", "superb", "superior", "sweet", "terrific", "thrilling", "touching",
    "triumph", "unforgettable", "vibrant", "warm", "warmth", "winner", "wonderful", "worthy"
  ]);

  const negativeWords = new Set([
    "abysmal", "annoying", "appalling", "atrocious", "awful", "bad", "banal", "bizarre",
    "bland", "boring", "cheap", "cheesy", "cliche", "cliched", "clumsy", "confusing",
    "contrived", "crap", "cringe", "crud", "damaged", "dead", "deficient", "deplorable",
    "depressing", "derivative", "disagreeable", "disappointing", "disappointment", "disaster",
    "disastrous", "disgusting", "dislike", "disliked", "dislikes", "dismal", "dreadful",
    "dull", "embarrassing", "empty", "fail", "failed", "failure", "fake", "flawed", "flaws",
    "flop", "forced", "forgettable", "garbage", "generic", "gross", "halfhearted", "hate",
    "hated", "hater", "hates", "hating", "hollow", "hopeless", "horrible", "horrid",
    "idiotic", "illogical", "incoherent", "incompetent", "insipid", "irritating", "junk",
    "lame", "lazy", "lifeless", "loathe", "loathed", "mediocre", "mess", "messy", "mindless",
    "miserable", "misguided", "monotonous", "negative", "nonsense", "obnoxious", "offensive",
    "painful", "pathetic", "pointless", "poor", "poorly", "predictable", "pretentious",
    "problematic", "ridiculous", "rubbish", "ruined", "shallow", "shame", "shoddy", "silly",
    "sloppy", "slow", "stale", "sterile", "stilted", "stupid", "superficial", "tedious",
    "terrible", "tiring", "trash", "trite", "unappealing", "unattractive", "unbearable",
    "unconvincing", "unengaging", "unenjoyable", "unfocused", "unfunny", "uninspired",
    "uninteresting", "unpleasant", "unwatchable", "vapid", "waste", "wasted", "weak",
    "wearisome", "woeful", "worst", "worthless", "wretched"
  ]);

  const negators = new Set([
    "no", "not", "never", "neither", "nor", "hardly", "scarcely", "barely",
    "cannot", "cant", "dont", "didnt", "isnt", "arent", "wasnt", "werent",
    "wont", "wouldnt", "couldnt", "shouldnt", "hasnt", "havent", "hadnt"
  ]);

  const initial = `<p class="model-tag">BASELINE STATUS</p><h3>Ready for a local analysis.</h3><p>It counts a small, documented set of positive and negative review terms. Its score is evidence from that lexicon, not a calibrated probability or neural-model confidence.</p><small>The LSTM and GRU remain documented research models; neither runs in this browser until validated ONNX artifacts are added.</small>`;

  const updateCounter = () => {
    counter.textContent = `${review.value.length.toLocaleString()} / 2,000`;
  };

  const analyze = () => {
    const text = review.value.trim();

    if (!text) {
      result.innerHTML = `<p class="model-tag">INPUT REQUIRED</p><h3>Add a review first.</h3><p>Please enter an English movie review sentence, then run the local baseline.</p>`;
      review.focus();
      return;
    }

    if (text.length > maxLength) {
      result.innerHTML = `<p class="model-tag">REVIEW TOO LONG</p><h3>Use ${maxLength.toLocaleString()} characters or fewer.</h3><p>Please shorten the review text to run the lexical baseline.</p>`;
      review.focus();
      return;
    }

    result.innerHTML = `<p class="model-tag">ANALYZING LOCALLY</p><h3>Reading lexical cues…</h3><p>No review text leaves this browser.</p>`;

    window.setTimeout(() => {
      // Normalize contractions (e.g. don't -> dont, isn't -> isnt)
      const normalized = text
        .toLowerCase()
        .replace(/n['’]t\b/g, "nt")
        .replace(/['’]/g, "");

      const tokens = normalized.match(/[a-z]+/g) || [];
      let score = 0;
      const matched = [];
      let negationWindow = 0;

      tokens.forEach((token) => {
        if (negators.has(token)) {
          negationWindow = 3;
          return;
        }

        let polarity = 0;
        if (positiveWords.has(token)) polarity = 1;
        else if (negativeWords.has(token)) polarity = -1;

        if (polarity !== 0) {
          const isNegated = negationWindow > 0;
          const adjusted = isNegated ? -polarity : polarity;
          score += adjusted;
          matched.push({
            display: `${isNegated ? "not " : ""}${token}`,
            adjusted
          });
          negationWindow = 0;
        } else if (negationWindow > 0) {
          negationWindow -= 1;
        }
      });

      let sentiment;
      if (score > 0) {
        sentiment = "Positive";
      } else if (score < 0) {
        sentiment = "Negative";
      } else {
        sentiment = "Neutral / Uncertain";
      }

      const evidence = matched.length;
      let detail;
      if (evidence > 0) {
        const scoreStr = score > 0 ? `+${score}` : `${score}`;
        const termsStr = matched.map((m) => `<code>${m.display}</code>`).join(" ");
        detail = `Lexicon score: <b>${scoreStr}</b> from ${evidence} matched ${evidence === 1 ? "term" : "terms"} (${termsStr}).`;
      } else {
        detail = "No decisive sentiment terms were detected in the lexicon.";
      }

      const note = score === 0
        ? "The lexicon baseline cannot determine a direction without decisive keywords. Deep contextual models (LSTM/GRU) are designed to handle such nuances."
        : "This is a simple browser lexicon tally, not a confidence score and not an LSTM or GRU prediction.";

      result.innerHTML = `<p class="model-tag">LOCAL LEXICON BASELINE</p><h3>${sentiment}</h3><p>${detail}</p><small>${note}</small>`;
    }, 120);
  };

  review.addEventListener("input", updateCounter);
  document.getElementById("reset").addEventListener("click", () => {
    review.value = "";
    updateCounter();
    result.innerHTML = initial;
    review.focus();
  });
  document.getElementById("example").addEventListener("click", () => {
    review.value = example;
    updateCounter();
    review.focus();
  });
  document.getElementById("predict").addEventListener("click", analyze);
})();
