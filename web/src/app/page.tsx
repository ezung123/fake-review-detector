"use client";

import React, { useState, useEffect } from "react";
import {
  AnalyzeResponse,
  HealthResponse,
  VerdictType,
  TokenExplanation,
} from "../types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const SAMPLE_REVIEWS = [
  {
    label: "Sample 1: Genuine Human",
    tag: "Authentic Nuance",
    text: "We stayed for three nights during the marathon weekend. Room 402 had a nice view of the park, though the street noise was audible until about 11 PM. The water pressure in the shower was a bit weak and the elevator took forever during checkout, but the front desk clerk kindly helped us hail a taxi in the rain.",
  },
  {
    label: "Sample 2: AI-Generated",
    tag: "Synthetic Polish",
    text: "Nestled in the vibrant heart of the city, this magnificent establishment stands as a true testament to world-class hospitality. From the breathtaking architectural design to the impeccably curated guest suites, every moment unfolds seamlessly. The culinary offerings delve into an exquisite tapestry of flavors. A quintessential experience for discerning travelers.",
  },
  {
    label: "Sample 3: Deceptive Human",
    tag: "Superlative Spam",
    text: "Absolutely the best hotel in the entire universe! The staff treated us like absolute royalty and gave us free champagne every single hour. Everything was pure perfection and flawless from start to finish. You must visit this heavenly place immediately, do not even consider any other hotel in this city!",
  },
];

const VERDICT_DETAILS: Record<
  VerdictType,
  { title: string; color: string; badgeBg: string; borderColor: string; description: string }
> = {
  genuine_human: {
    title: "Likely genuine, human-written",
    color: "text-emerald-400",
    badgeBg: "bg-emerald-500/10 text-emerald-300 border-emerald-500/30",
    borderColor: "border-emerald-500/40",
    description: "The text displays organic human linguistic variance, sensory specificities, and balanced experiential nuance.",
  },
  deceptive_human: {
    title: "Likely deceptive, human-written",
    color: "text-amber-400",
    badgeBg: "bg-amber-500/10 text-amber-300 border-amber-500/30",
    borderColor: "border-amber-500/40",
    description: "Human authorship characteristics detected, but with strong indicators of promotional manipulation, inflated superlatives, or deceptive intent.",
  },
  ai_genuine: {
    title: "Likely AI-written, no sign of deception",
    color: "text-cyan-400",
    badgeBg: "bg-cyan-500/10 text-cyan-300 border-cyan-500/30",
    borderColor: "border-cyan-500/40",
    description: "Synthetically structured or LLM-assisted phrasing, yet expressing realistic, non-manipulative product feedback.",
  },
  ai_deceptive: {
    title: "Likely AI-generated and deceptive",
    color: "text-rose-400",
    badgeBg: "bg-rose-500/10 text-rose-300 border-rose-500/30",
    borderColor: "border-rose-500/40",
    description: "High probability of automated generation combined with deceptive marketing spam and artificial sentiment inflation.",
  },
};

export default function HomePage() {
  const [reviewText, setReviewText] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [activeTab, setActiveTab] = useState<"authorship" | "deception">("authorship");
  const [serverMode, setServerMode] = useState<string>("stub");
  const [apiConnected, setApiConnected] = useState<boolean>(false);

  // Probe backend health status on mount
  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (res.ok) {
          const data: HealthResponse = await res.json();
          setServerMode(data.mode);
          setApiConnected(true);
        } else {
          setApiConnected(false);
        }
      } catch {
        setApiConnected(false);
        setServerMode("stub");
      }
    }
    checkHealth();
  }, []);

  const charCount = reviewText.length;
  const isTooShort = charCount > 0 && charCount < 20;
  const isTooLong = charCount > 3000;
  const isValidLength = charCount >= 20 && charCount <= 3000;

  async function handleAnalyze() {
    if (!isValidLength || isLoading) return;
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: reviewText }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server responded with error ${response.status}`);
      }

      const data: AnalyzeResponse = await response.json();
      setAnalysis(data);
      setServerMode(data.model_info.mode);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to connect to the backend API.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }

  function handleSelectSample(text: string) {
    setReviewText(text);
    setError(null);
  }

  // Render review text with inline token highlights
  function renderHighlightedText(text: string, tokens: TokenExplanation[]) {
    if (!tokens || tokens.length === 0) return text;

    // Create lookup map of lowercase token to weight
    const tokenMap = new Map<string, number>();
    for (const t of tokens) {
      tokenMap.set(t.token.toLowerCase(), t.weight);
    }

    // Split text keeping punctuation intact
    const parts = text.split(/(\b[a-zA-Z]{3,}\b)/g);

    return parts.map((part, index) => {
      const lower = part.toLowerCase();
      const weight = tokenMap.get(lower);

      if (weight !== undefined) {
        const isPositive = weight > 0;
        // Positive weight pushes towards AI / Deceptive (red/warm)
        // Negative weight pushes towards Human / Genuine (green/cool)
        const bgClass = isPositive
          ? "bg-rose-500/25 text-rose-200 border-b-2 border-rose-400 font-medium px-1 rounded-sm"
          : "bg-emerald-500/25 text-emerald-200 border-b-2 border-emerald-400 font-medium px-1 rounded-sm";

        return (
          <span
            key={index}
            className={`inline-block relative group transition-all duration-150 ${bgClass}`}
            title={`${isPositive ? "+" : ""}${weight.toFixed(3)} (${
              activeTab === "authorship"
                ? isPositive ? "Pushes toward AI" : "Pushes toward Human"
                : isPositive ? "Pushes toward Deceptive" : "Pushes toward Genuine"
            })`}
          >
            {part}
            <span className="sr-only">({weight})</span>
          </span>
        );
      }
      return <span key={index}>{part}</span>;
    });
  }

  return (
    <div className="min-h-screen text-slate-100 antialiased flex flex-col justify-between">
      {/* Demo Mode Banner (shown when models are in stub mode) */}
      {(serverMode === "stub" || !apiConnected) && (
        <div className="bg-gradient-to-r from-amber-600/90 via-amber-500/90 to-yellow-600/90 text-slate-950 font-semibold px-4 py-2.5 text-center text-sm shadow-md flex items-center justify-center gap-2">
          <svg className="w-4 h-4 fill-current" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
          <span>Demo mode: models not loaded</span>
          <span className="text-xs opacity-85 font-normal">
            ({apiConnected ? "API running in deterministic heuristic stub mode" : "Backend unreachable; ensure FastAPI is running on port 8000"})
          </span>
        </div>
      )}

      {/* Main Container */}
      <main className="max-w-6xl w-full mx-auto px-4 py-8 sm:px-6 lg:px-8 space-y-8 flex-1">
        {/* Header */}
        <header className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
            M.Sc. Data Science Minor Project
          </div>
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-200 to-indigo-300 bg-clip-text text-transparent">
            Detection of Fake and AI-Generated Reviews
          </h1>
          <p className="text-base sm:text-lg text-slate-400 max-w-3xl mx-auto">
            A combined approach for e-commerce and hospitality trust & safety. Evaluating reviews across two independent, orthogonal dimensions: <strong className="text-slate-200">Authorship</strong> and <strong className="text-slate-200">Deception</strong>.
          </p>
          <p className="text-xs text-slate-500 italic">
            * Note: All assessment outputs are probabilistic model estimates, not definitive legal proof.
          </p>
        </header>

        {/* Quick Sample Selector */}
        <section className="bg-slate-900/60 backdrop-blur border border-slate-800 rounded-xl p-4 sm:p-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
            <span className="text-xs uppercase tracking-wider font-semibold text-slate-400">
              Quick Test Sample Reviews:
            </span>
            <span className="text-xs text-slate-500">
              Click any sample to populate the input below
            </span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {SAMPLE_REVIEWS.map((sample, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => handleSelectSample(sample.text)}
                className="text-left p-3.5 rounded-lg bg-slate-800/60 hover:bg-slate-800 hover:border-indigo-500/40 border border-slate-700/60 transition-all duration-150 group"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-semibold text-slate-200 group-hover:text-indigo-300 transition-colors">
                    {sample.label}
                  </span>
                  <span className="text-[10px] uppercase font-bold px-1.5 py-0.5 rounded bg-slate-700 text-slate-300">
                    {sample.tag}
                  </span>
                </div>
                <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed">
                  {sample.text}
                </p>
              </button>
            ))}
          </div>
        </section>

        {/* Input Textarea & Analyze Action */}
        <section className="bg-slate-900/80 backdrop-blur border border-slate-800 rounded-2xl p-5 sm:p-6 shadow-xl space-y-4">
          <div className="flex items-center justify-between">
            <label htmlFor="review-input" className="block text-sm font-medium text-slate-200">
              Paste Review Text
            </label>
            <div className="flex items-center gap-2 text-xs">
              <span
                className={`font-mono ${
                  isTooShort || isTooLong ? "text-amber-400 font-semibold" : "text-slate-400"
                }`}
              >
                {charCount} / 3000 chars
              </span>
              {isTooShort && (
                <span className="text-amber-400">(minimum 20 characters required)</span>
              )}
              {isTooLong && (
                <span className="text-rose-400">(maximum 3000 characters exceeded)</span>
              )}
            </div>
          </div>

          <div className="relative">
            <textarea
              id="review-input"
              rows={5}
              value={reviewText}
              onChange={(e) => setReviewText(e.target.value)}
              placeholder="Paste a hotel, product, or restaurant review here (minimum 20 characters)..."
              className="w-full bg-slate-950/80 border border-slate-700/80 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 rounded-xl p-4 text-sm text-slate-100 placeholder-slate-500 transition-all outline-none resize-y"
            />
          </div>

          {error && (
            <div className="p-3.5 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm flex items-start gap-2">
              <svg className="w-5 h-5 shrink-0 text-rose-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <span>{error}</span>
            </div>
          )}

          <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-1">
            <div className="text-xs text-slate-500">
              Evaluates both Authorship (AI vs. Human) &amp; Deception (Deceptive vs. Genuine)
            </div>
            <button
              type="button"
              onClick={handleAnalyze}
              disabled={!isValidLength || isLoading}
              className={`w-full sm:w-auto px-6 py-2.5 rounded-xl font-medium text-sm flex items-center justify-center gap-2 transition-all duration-150 ${
                isValidLength && !isLoading
                  ? "bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-600/30 active:scale-[0.98]"
                  : "bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/50"
              }`}
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  <span>Analyzing review...</span>
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z" />
                  </svg>
                  <span>Analyze Review</span>
                </>
              )}
            </button>
          </div>
        </section>

        {/* Results Presentation Section */}
        {analysis && (
          <section className="space-y-6 animate-fadeIn">
            {/* Verdict Card */}
            {(() => {
              const details = VERDICT_DETAILS[analysis.verdict];
              return (
                <div
                  className={`bg-slate-900/90 backdrop-blur border-2 ${details.borderColor} rounded-2xl p-6 shadow-2xl relative overflow-hidden`}
                >
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div className="space-y-1.5">
                      <div className="flex items-center gap-2">
                        <span className="text-xs uppercase tracking-wider font-semibold text-slate-400">
                          Combiner Verdict
                        </span>
                        <span
                          className={`text-xs px-2.5 py-0.5 rounded-full border font-semibold ${details.badgeBg}`}
                        >
                          {analysis.verdict}
                        </span>
                      </div>
                      <h2 className={`text-2xl sm:text-3xl font-extrabold ${details.color}`}>
                        {details.title}
                      </h2>
                      <p className="text-sm text-slate-300 max-w-2xl">
                        {details.description}
                      </p>
                    </div>

                    {/* 2x2 Quadrant Indicator */}
                    <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800 text-xs shrink-0 self-start md:self-auto">
                      <div className="text-[10px] uppercase font-bold text-slate-400 mb-2 text-center">
                        2D Decision Matrix
                      </div>
                      <div className="grid grid-cols-2 gap-1.5 text-center text-[11px] font-mono">
                        <div
                          className={`px-2 py-1 rounded border ${
                            analysis.verdict === "genuine_human"
                              ? "bg-emerald-500/20 border-emerald-500 text-emerald-300 font-bold"
                              : "bg-slate-900/40 border-slate-800 text-slate-600"
                          }`}
                        >
                          Human + Genuine
                        </div>
                        <div
                          className={`px-2 py-1 rounded border ${
                            analysis.verdict === "deceptive_human"
                              ? "bg-amber-500/20 border-amber-500 text-amber-300 font-bold"
                              : "bg-slate-900/40 border-slate-800 text-slate-600"
                          }`}
                        >
                          Human + Deceptive
                        </div>
                        <div
                          className={`px-2 py-1 rounded border ${
                            analysis.verdict === "ai_genuine"
                              ? "bg-cyan-500/20 border-cyan-500 text-cyan-300 font-bold"
                              : "bg-slate-900/40 border-slate-800 text-slate-600"
                          }`}
                        >
                          AI + Genuine
                        </div>
                        <div
                          className={`px-2 py-1 rounded border ${
                            analysis.verdict === "ai_deceptive"
                              ? "bg-rose-500/20 border-rose-500 text-rose-300 font-bold"
                              : "bg-slate-900/40 border-slate-800 text-slate-600"
                          }`}
                        >
                          AI + Deceptive
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })()}

            {/* Two Independent Probability Gauges */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {/* Authorship Dimension */}
              <div className="bg-slate-900/80 backdrop-blur border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-indigo-400" />
                    <h3 className="font-semibold text-slate-200 text-base">
                      (1) Authorship Dimension
                    </h3>
                  </div>
                  <span
                    className={`text-xs px-2 py-0.5 rounded font-mono font-bold ${
                      analysis.authorship.label === "ai"
                        ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                        : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                    }`}
                  >
                    Predicted: {analysis.authorship.label.toUpperCase()}
                  </span>
                </div>

                <p className="text-xs text-slate-400">
                  Measures linguistic artifacts to determine if text was generated by an LLM vs written by a human.
                </p>

                {/* Probability Bar */}
                <div className="space-y-1.5 pt-1">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-emerald-400">
                      Human ({(100 - analysis.authorship.ai_probability * 100).toFixed(1)}%)
                    </span>
                    <span className="text-rose-400">
                      AI ({(analysis.authorship.ai_probability * 100).toFixed(1)}%)
                    </span>
                  </div>
                  <div className="w-full bg-slate-950 h-3 rounded-full overflow-hidden p-0.5 border border-slate-800">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-emerald-500 via-amber-400 to-rose-500 transition-all duration-500"
                      style={{ width: `${Math.max(analysis.authorship.ai_probability * 100, 3)}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-[10px] text-slate-500">
                    <span>Threshold: 0.50</span>
                    <span>Score: {analysis.authorship.ai_probability.toFixed(3)}</span>
                  </div>
                </div>
              </div>

              {/* Deception Dimension */}
              <div className="bg-slate-900/80 backdrop-blur border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-fuchsia-400" />
                    <h3 className="font-semibold text-slate-200 text-base">
                      (2) Deception Dimension
                    </h3>
                  </div>
                  <span
                    className={`text-xs px-2 py-0.5 rounded font-mono font-bold ${
                      analysis.deception.label === "deceptive"
                        ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                        : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                    }`}
                  >
                    Predicted: {analysis.deception.label.toUpperCase()}
                  </span>
                </div>

                <p className="text-xs text-slate-400">
                  Measures sentiment exaggeration, lack of concrete details, and spam hallmarks to detect fraud.
                </p>

                {/* Probability Bar */}
                <div className="space-y-1.5 pt-1">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-emerald-400">
                      Genuine ({(100 - analysis.deception.deceptive_probability * 100).toFixed(1)}%)
                    </span>
                    <span className="text-rose-400">
                      Deceptive ({(analysis.deception.deceptive_probability * 100).toFixed(1)}%)
                    </span>
                  </div>
                  <div className="w-full bg-slate-950 h-3 rounded-full overflow-hidden p-0.5 border border-slate-800">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-emerald-500 via-amber-400 to-rose-500 transition-all duration-500"
                      style={{ width: `${Math.max(analysis.deception.deceptive_probability * 100, 3)}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-[10px] text-slate-500">
                    <span>Threshold: 0.50</span>
                    <span>Score: {analysis.deception.deceptive_probability.toFixed(3)}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Word-Highlight & Explanation Area */}
            <div className="bg-slate-900/80 backdrop-blur border border-slate-800 rounded-2xl p-5 sm:p-6 shadow-xl space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
                <div>
                  <h3 className="text-base font-semibold text-slate-200">
                    Word-Level Explanation Highlights
                  </h3>
                  <p className="text-xs text-slate-400">
                    Token attribution weights showing which words pushed the model toward or away from each class.
                  </p>
                </div>

                {/* Toggle dimension */}
                <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 self-start sm:self-auto">
                  <button
                    type="button"
                    onClick={() => setActiveTab("authorship")}
                    className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                      activeTab === "authorship"
                        ? "bg-indigo-600 text-white shadow"
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    Authorship Tokens
                  </button>
                  <button
                    type="button"
                    onClick={() => setActiveTab("deception")}
                    className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                      activeTab === "deception"
                        ? "bg-indigo-600 text-white shadow"
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    Deception Tokens
                  </button>
                </div>
              </div>

              {/* Legend */}
              <div className="flex flex-wrap items-center gap-4 text-xs text-slate-400 bg-slate-950/50 p-3 rounded-lg border border-slate-800/80">
                <span className="font-semibold text-slate-300">Color Legend:</span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded bg-rose-500/40 border border-rose-400" />
                  <span>
                    Positive weight (&gt; 0): Pushes toward{" "}
                    <strong className="text-rose-300">
                      {activeTab === "authorship" ? "AI" : "Deceptive"}
                    </strong>
                  </span>
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded bg-emerald-500/40 border border-emerald-400" />
                  <span>
                    Negative weight (&lt; 0): Pushes toward{" "}
                    <strong className="text-emerald-300">
                      {activeTab === "authorship" ? "Human" : "Genuine"}
                    </strong>
                  </span>
                </span>
              </div>

              {/* Inline highlighted text view */}
              <div className="space-y-2">
                <div className="text-xs uppercase font-semibold text-slate-400">
                  Review Text with Inline Feature Attributions:
                </div>
                <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 text-sm text-slate-200 leading-relaxed max-h-64 overflow-y-auto">
                  {renderHighlightedText(
                    reviewText,
                    activeTab === "authorship"
                      ? analysis.explanations.authorship
                      : analysis.explanations.deception
                  )}
                </div>
              </div>

              {/* Token Attribution Badges */}
              <div className="space-y-2">
                <div className="text-xs uppercase font-semibold text-slate-400">
                  Salient Feature Tokens Ranked by Attribution Weight:
                </div>
                <div className="flex flex-wrap gap-2">
                  {(activeTab === "authorship"
                    ? analysis.explanations.authorship
                    : analysis.explanations.deception
                  ).map((item, idx) => {
                    const isPos = item.weight > 0;
                    return (
                      <span
                        key={idx}
                        className={`inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-md border font-mono ${
                          isPos
                            ? "bg-rose-500/10 text-rose-300 border-rose-500/30"
                            : "bg-emerald-500/10 text-emerald-300 border-emerald-500/30"
                        }`}
                      >
                        <span className="font-semibold font-sans">{item.token}</span>
                        <span className="text-[10px] opacity-80">
                          {isPos ? "+" : ""}
                          {item.weight.toFixed(3)}
                        </span>
                      </span>
                    );
                  })}
                </div>
              </div>
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 py-6 px-4 text-center text-xs text-slate-500 space-y-1">
        <p>
          Detection of Fake and AI-Generated Reviews &bull; M.Sc. Data Science Minor Project
        </p>
        <p>
          Authorship and Deception are preserved as separate orthogonal outputs.
        </p>
      </footer>
    </div>
  );
}

