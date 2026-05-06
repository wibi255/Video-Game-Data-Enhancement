# Video Game Data Enhancement

Enrich video game datasets using the Google AI Studio (Gemini) API with an automatic DeepSeek API fallback.

## Core Features

* **Optimization:** Batch processing (10 items/request).
* **Speed:** Parallel execution (10 workers) via ThreadPoolExecutor.
* **Accuracy:** Strict ID-based matching to guarantee original row sequence.
* **Reliability:** 3 retry attempts per batch with exponential backoff.
* **Smart Fallback:** Automatically switches to DeepSeek if Gemini fails.

## Setup & Configuration

1. Install Dependencies
    Ensure you have Python installed, then run:

    pip install -r requirements.txt

2. Set API Keys
    Create a .env file in the project root folder and add your keys:

    GEMINI_API_KEY=your_gemini_key
    DEEPSEEK_API_KEY=your_deepseek_key

3. Prepare Input Data
    Save your input file as data/game-thumbnail.csv. It must include these headers:

    game_title,image_url
    Street Fighter 6,[https://images.igdb.com/](https://images.igdb.com/)...
    Hunt: Showdown 1896,[https://images.igdb.com/](https://images.igdb.com/)...
    Usage
    Run the script via terminal:

    python main.py
    Expected Output
    The script generates a new file at data/Enhanced_Game_Data.csv containing your original data plus these new columns:

    genre: Single-word genre (e.g., Shooter, RPG).

    short_description: Brief summary (max 30 words).

    player_mode: Singleplayer / Multiplayer / Both.

    api_source: The API that successfully processed the row (gemini / deepseek).