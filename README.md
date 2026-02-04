# Synthesis — AI-Powered Career Exploration

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?logo=googlegemini&logoColor=white)
![Claude](https://img.shields.io/badge/Claude_Opus_4-CC785C?logo=anthropic&logoColor=white)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://synthesis-web.streamlit.app/)
![Opik](https://img.shields.io/badge/Opik-Tracing-00D4AA?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgZmlsbD0iIzAwRDRBQSIvPjwvc3ZnPg==)
![Encode Hackathon](https://img.shields.io/badge/Encode_Hackathon-2026-FF6B6B)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

An interactive multi-agent application that helps you discover career paths, create personalized learning plans, and connect with relevant communities.

> Adapted from [Recipe Agent Demo with Google ADK and Opik](https://gist.github.com/vincentkoc/638f11fcb00d0351473be5a8705d4c08).


## Prerequisites

- **Pixi** — Package manager (dependencies defined in `pixi.toml`)
- **Python 3.10+**
- **API Keys:**
  - Google Gemini API key (`GOOGLE_API_KEY`) — required for CLI and streamlit app
  - Opik API key (`OPIK_API_KEY`) — required for tracing
  - OpenRouter API key (`OPENROUTER_API_KEY`) — for prompt optimization
  - *Or* OpenAI API key (`OPENAI_API_KEY`) — alternative for prompt optimization
  - Set the API keys in `.env`
    ```bash
    GOOGLE_API_KEY=your_google_api_key
    OPENROUTER_API_KEY=your_openrouter_api_key
    OPIK_API_KEY=your_opik_api_key
    OPIK_WORKSPACE=your_opik_workspace
    ```

## Getting Started

### Basic Usage
```bash
# Run the interactive CLI app
pixi run cli

# Run the Streamlit web app
pixi run web

# Clean generated files (results, project plans, optimized prompts)
pixi run clean
```

### Advanced Usage
```bash
# Run evaluation metrics on agent outputs
pixi run eval

# Optimize agent prompts (requires OPENROUTER_API_KEY)
pixi run opt-role   # Role generation prompt
pixi run opt-learn  # Learning path prompt
pixi run opt-do     # Test generation prompt
pixi run opt-go     # Community/project prompt
```

## ⚠️ Important Notice

This application uses AI models to generate recommendations, learning paths, and career suggestions. While we strive for accuracy:

- **AI can make mistakes.** Always verify recommendations with trusted sources.
- **Do your own research.** Treat AI outputs as a starting point, not a final answer.
- **Community links may be outdated.** Verify that resources and communities still exist.

## Disclaimer

Built for **Encode Hackathon 2026**. Provided as-is with no warranty. Use at your own discretion.

## License

This project is licensed under the [MIT License](LICENSE).