// Type stubs for optional / 3rd-party modules used only in certain environments

declare module '@google/genai' {
  export class GoogleGenAI {
    constructor(opts?: any);
    models: any;
  }
}

// Provide a very small 'process' declaration so code referencing process.env at build-time
// doesn't fail static checks in environments that don't supply node types.
declare var process: any;
