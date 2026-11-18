export type AuthorshipLabel = "human" | "ai";
export type DeceptionLabel = "genuine" | "deceptive";
export type VerdictType = "genuine_human" | "deceptive_human" | "ai_genuine" | "ai_deceptive";

export interface TokenExplanation {
  token: string;
  weight: number;
}

export interface Explanations {
  authorship: TokenExplanation[];
  deception: TokenExplanation[];
}

export interface AnalyzeResponse {
  authorship: {
    ai_probability: number;
    label: AuthorshipLabel;
  };
  deception: {
    deceptive_probability: number;
    label: DeceptionLabel;
  };
  verdict: VerdictType;
  explanations: Explanations;
  model_info: {
    mode: "stub" | "classical" | "transformer" | string;
  };
  warnings: string[];
}

export interface HealthResponse {
  status: "ok";
  mode: "stub" | "classical" | "transformer";
}

