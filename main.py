"""
=============================================================================
VIDEO GAME DATA ENHANCEMENT SCRIPT
=============================================================================
Task: Concise Video Game Data Enhancement with Google AI Studio API
      with DeepSeek API Fallback
Features: Batch processing (10 items/request), Parallel execution (10 workers),
          Strict ID-based matching to prevent out-of-order data corruption.
Date: 2026-05-06
=============================================================================
"""

import pandas as pd
import os
import time
import json
import re
import logging
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# CONFIGURATION & LOGGING SETUP
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("game_enhancement.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

CONFIG = {
    "INPUT_FILE": "data/game-thumbnail.csv",
    "OUTPUT_FILE": "data/Enhanced_Game_Data.csv",
    "GEMINI_MODEL": "gemini-2.5-flash",
    "DEEPSEEK_MODEL": "deepseek-v4-flash",
    "BATCH_SIZE": 10,
    "MAX_WORKERS": 10,
    "RATE_LIMIT_DELAY": 2,
    "MAX_RETRIES": 3,
    "RETRY_DELAY": 5,
    "MAX_DESCRIPTION_WORDS": 30,
    "VALID_PLAYER_MODES": ["Singleplayer", "Multiplayer", "Both"]
}

# ==========================================
# DATA CLASS
# ==========================================

@dataclass
class GameInfo:
    id: int  # Added explicit ID for bulletproof sorting and mapping
    game_title: str
    genre: str
    short_description: str
    player_mode: str
    source: str = "gemini"
    success: bool = True
    error_message: Optional[str] = None


# ==========================================
# 1. API SETUP
# ==========================================

def load_env_keys():
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY not found in .env")
    if not deepseek_key:
        raise ValueError("DEEPSEEK_API_KEY not found in .env")
    return gemini_key, deepseek_key


# ==========================================
# 2. BATCH PROMPT ENGINEERING
# ==========================================

def build_batch_prompt(game_items: List[Dict[str, Any]]) -> str:
    """Build prompt for batch of games, using IDs to guarantee sequence order."""
    games_list = "\n".join([f"- ID: {item['id']} | Title: {item['title']}" for item in game_items])

    return (
        f"You are a video game database expert. Analyze the following {len(game_items)} games.\n\n"
        f"TASK: Provide information for ALL games in this EXACT JSON array format (no markdown, no extra text):\n\n"
        f"[\n"
        f'  {{"id": EXACT_ID_NUMBER, "genre": "ONE_WORD_GENRE", "short_description": "Brief description under 30 words", "player_mode": "Singleplayer" or "Multiplayer" or "Both"}},\n'
        f"  ...\n"
        f"]\n\n"
        f"RULES:\n"
        f"- Return a JSON ARRAY with exactly {len(game_items)} objects.\n"
        f"- Each object MUST contain the exact 'id' integer provided in the input list.\n"
        f"- genre examples: Shooter, RPG, Simulation, Strategy, Action, Adventure, Fighting, Survival, MOBA, Battle-Royale\n"
        f"- player_mode ONLY: \"Singleplayer\", \"Multiplayer\", or \"Both\"\n"
        f"- Output ONLY the JSON array, nothing else. Do not wrap in ```json.\n\n"
        f"GAMES TO ANALYZE:\n"
        f"{games_list}\n\n"
        f"OUTPUT:"
    )


# ==========================================
# 3. RESPONSE CLEANING & PARSING
# ==========================================

def clean_json_response(text: str) -> str:
    text = text.strip()
    
    # Perbaikan: String literal digabung menjadi satu baris
    if text.startswith("```"):
        parts = text.split("```")
        for part in parts:
            clean_part = part.strip()
            # Mencari bagian yang berisi data, bukan sekadar tag 'json'
            if clean_part and not clean_part.lower().startswith("json"):
                text = clean_part
                break
    else:
        # Mencari pola array JSON jika tidak ada tag markdown
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
            
    # Membersihkan awalan kata 'json' jika masih ada yang tertinggal
    text = re.sub(r"^json\s*", "", text, flags=re.IGNORECASE).strip()
    
    return text


def parse_single_game(data: Dict[str, Any], expected_id: int, expected_title: str) -> GameInfo:
    try:
        genre = str(data.get("genre", "Unknown")).strip()
        description = str(data.get("short_description", "No description")).strip()
        player_mode = str(data.get("player_mode", "Unknown")).strip()

        genre = genre.split()[0] if genre else "Unknown"

        if player_mode not in CONFIG["VALID_PLAYER_MODES"]:
            player_mode = "Both"

        words = description.split()
        if len(words) > CONFIG["MAX_DESCRIPTION_WORDS"]:
            description = " ".join(words[:CONFIG["MAX_DESCRIPTION_WORDS"]]) + "..."

        return GameInfo(
            id=expected_id,
            game_title=expected_title,
            genre=genre,
            short_description=description,
            player_mode=player_mode,
            success=True
        )
    except Exception as e:
        return GameInfo(
            id=expected_id,
            game_title=expected_title,
            genre="Error",
            short_description="Error",
            player_mode="Error",
            success=False,
            error_message=str(e)
        )


def parse_batch_response(raw_text: str, expected_items: List[Dict[str, Any]], source: str) -> List[GameInfo]:
    results = []
    try:
        cleaned = clean_json_response(raw_text)
        logger.debug(f"Cleaned batch response: {cleaned[:500]}")
        data_list = json.loads(cleaned)

        if not isinstance(data_list, list):
            raise ValueError(f"Expected JSON array, got {type(data_list)}")

        # Create a dictionary map by ID to prevent out-of-order errors
        id_to_result = {}
        for item in data_list:
            if isinstance(item, dict) and "id" in item:
                try:
                    item_id = int(item["id"])
                    id_to_result[item_id] = item
                except (ValueError, TypeError):
                    continue

        for exp_item in expected_items:
            exp_id = exp_item["id"]
            exp_title = exp_item["title"]

            if exp_id in id_to_result:
                info = parse_single_game(id_to_result[exp_id], exp_id, exp_title)
                info.source = source
                results.append(info)
            else:
                logger.warning(f"ID {exp_id} ('{exp_title}') not found in response.")
                results.append(GameInfo(
                    id=exp_id, game_title=exp_title, genre="Error", short_description="Error",
                    player_mode="Error", source=source, success=False,
                    error_message="ID not found in API response array"
                ))

        return results

    except Exception as e:
        logger.error(f"Parse error: {e}")
        for exp_item in expected_items:
            results.append(GameInfo(
                id=exp_item["id"], game_title=exp_item["title"], genre="Error",
                short_description="Error", player_mode="Error", source=source,
                success=False, error_message=f"JSON/Parse error: {str(e)}"
            ))
        return results


# ==========================================
# 4. GEMINI BATCH API CALL
# ==========================================

def call_gemini_batch(api_key: str, game_items: List[Dict[str, Any]]) -> List[GameInfo]:
    prompt = build_batch_prompt(game_items)

    for attempt in range(1, CONFIG["MAX_RETRIES"] + 1):
        try:
            logger.info(f"[Gemini] Batch of {len(game_items)} games (Attempt {attempt}/{CONFIG['MAX_RETRIES']})")

            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(CONFIG["GEMINI_MODEL"])

            response = model.generate_content(prompt)
            results = parse_batch_response(response.text, game_items, "gemini")

            success_count = sum(1 for r in results if r.success)
            logger.info(f"[Gemini] Batch result: {success_count}/{len(game_items)} success")

            if success_count == len(game_items):
                return results
            if attempt < CONFIG["MAX_RETRIES"]:
                time.sleep(CONFIG["RETRY_DELAY"])

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[Gemini] Batch error (Attempt {attempt}): {error_msg}")
            if "429" in error_msg or "rate limit" in error_msg.lower():
                wait_time = CONFIG["RETRY_DELAY"] * attempt
                logger.warning(f"[Gemini] Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < CONFIG["MAX_RETRIES"]:
                time.sleep(CONFIG["RETRY_DELAY"])

    logger.error(f"[Gemini] All retries failed for batch")
    return [GameInfo(
        id=item["id"], game_title=item["title"], genre="Error", short_description="Error",
        player_mode="Error", source="gemini", success=False,
        error_message="Gemini batch max retries exceeded"
    ) for item in game_items]


# ==========================================
# 5. DEEPSEEK BATCH API CALL
# ==========================================

def call_deepseek_batch(api_key: str, game_items: List[Dict[str, Any]]) -> List[GameInfo]:
    prompt = build_batch_prompt(game_items)

    for attempt in range(1, CONFIG["MAX_RETRIES"] + 1):
        try:
            logger.info(f"[DeepSeek] Batch of {len(game_items)} games (Attempt {attempt}/{CONFIG['MAX_RETRIES']})")

            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

            response = client.chat.completions.create(
                model=CONFIG["DEEPSEEK_MODEL"],
                messages=[
                    {"role": "system", "content": "You are a highly precise video game database expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )

            raw_text = response.choices[0].message.content
            results = parse_batch_response(raw_text, game_items, "deepseek")

            success_count = sum(1 for r in results if r.success)
            logger.info(f"[DeepSeek] Batch result: {success_count}/{len(game_items)} success")

            if success_count == len(game_items):
                return results
            if attempt < CONFIG["MAX_RETRIES"]:
                time.sleep(CONFIG["RETRY_DELAY"])

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[DeepSeek] Batch error (Attempt {attempt}): {error_msg}")
            if "429" in error_msg or "rate limit" in error_msg.lower():
                wait_time = CONFIG["RETRY_DELAY"] * attempt
                logger.warning(f"[DeepSeek] Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < CONFIG["MAX_RETRIES"]:
                time.sleep(CONFIG["RETRY_DELAY"])

    logger.error(f"[DeepSeek] All retries failed for batch")
    return [GameInfo(
        id=item["id"], game_title=item["title"], genre="Error", short_description="Error",
        player_mode="Error", source="deepseek", success=False,
        error_message="DeepSeek batch max retries exceeded"
    ) for item in game_items]


# ==========================================
# 6. FALLBACK LOGIC FOR BATCH
# ==========================================

def process_batch_with_fallback(gemini_key: str, deepseek_key: str, game_items: List[Dict[str, Any]]) -> List[GameInfo]:
    """Try Gemini batch first, fallback to DeepSeek batch if fails."""
    results = call_gemini_batch(gemini_key, game_items)

    failed_indices = [i for i, r in enumerate(results) if not r.success]

    if not failed_indices:
        return results

    logger.warning(f"[Fallback] {len(failed_indices)} items failed on Gemini. Retrying with DeepSeek...")

    failed_items = [game_items[i] for i in failed_indices]
    fallback_results = call_deepseek_batch(deepseek_key, failed_items)

    for idx, fallback_idx in enumerate(failed_indices):
        results[fallback_idx] = fallback_results[idx]

    return results


# ==========================================
# 7. PARALLEL BATCH PROCESSING
# ==========================================

def process_all_batches_parallel(gemini_key: str, deepseek_key: str, all_titles: List[str]) -> List[GameInfo]:
    """Split into batches of 10, process up to 10 batches in parallel, and ensure order."""

    # Assign a unique ID to every item to guarantee ordering at the end
    all_items = [{"id": idx, "title": title} for idx, title in enumerate(all_titles)]

    batches = []
    for i in range(0, len(all_items), CONFIG["BATCH_SIZE"]):
        batch = all_items[i:i + CONFIG["BATCH_SIZE"]]
        batches.append(batch)

    logger.info(f"Total games: {len(all_items)}")
    logger.info(f"Total batches: {len(batches)} (batch size: {CONFIG['BATCH_SIZE']})")
    logger.info(f"Max parallel workers: {CONFIG['MAX_WORKERS']}")

    # Use an array to store results mapped to their exact batch index
    # This prevents ThreadPool from scrambling the data order as they complete randomly!
    ordered_batch_results = [None] * len(batches)
    completed_batches = 0
    total_batches = len(batches)

    with ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        future_to_batch = {
            executor.submit(process_batch_with_fallback, gemini_key, deepseek_key, batch): batch_idx
            for batch_idx, batch in enumerate(batches)
        }

        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            batch = batches[batch_idx]
            try:
                batch_results = future.result()
                ordered_batch_results[batch_idx] = batch_results
                completed_batches += 1

                success_in_batch = sum(1 for r in batch_results if r.success)
                print(f"[Batch {batch_idx + 1}/{total_batches}] Completed: {success_in_batch}/{len(batch_results)} success")
                print(f"  Progress: {completed_batches}/{total_batches} batches done")

            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} failed: {e}")
                ordered_batch_results[batch_idx] = [
                    GameInfo(id=item["id"], game_title=item["title"], genre="Error", 
                             short_description="Error", player_mode="Error", 
                             source="none", success=False, error_message=f"Batch execution failed: {str(e)}") 
                    for item in batch
                ]
                completed_batches += 1

    # Flatten the ordered results
    all_results = []
    for batch_res in ordered_batch_results:
        if batch_res:
            all_results.extend(batch_res)

    # Final absolute safety check: sort exactly by ID just in case
    all_results.sort(key=lambda x: x.id)

    return all_results


# ==========================================
# 8. CSV PROCESSING
# ==========================================

def load_csv(file_path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File '{file_path}' not found.\n"
            f"Expected structure:\n"
            f"  project/\n"
            f"  ├── main.py\n"
            f"  ├── data/\n"
            f"  │   └── game-thumbnail.csv\n"
            f"  └── .env"
        )
    df = pd.read_csv(file_path)
    required = ["game_title", "image_url"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df.columns = df.columns.str.strip()
    for col in ["game_title", "image_url"]:
        df[col] = df[col].astype(str).str.strip()
    logger.info(f"Loaded {len(df)} rows")
    return df


def save_results(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Saved to: {output_path}")
    logger.info(f"Total rows: {len(df)}")


# ==========================================
# 9. MAIN PIPELINE
# ==========================================

def main():
    print("=" * 60)
    print("VIDEO GAME DATA ENHANCEMENT (SECURE ORDERING)")
    print("Batch: 10 items/request | Parallel: 10 workers")
    print("Primary: Gemini API | Fallback: DeepSeek API")
    print("=" * 60)

    try:
        gemini_key, deepseek_key = load_env_keys()
        df = load_csv(CONFIG["INPUT_FILE"])

        all_titles = df["game_title"].tolist()
        logger.info(f"Processing {len(all_titles)} games...")

        start_time = time.time()
        results = process_all_batches_parallel(gemini_key, deepseek_key, all_titles)
        elapsed = time.time() - start_time

        # Data is now 100% aligned with df because we sorted by 'id'
        genres = [r.genre for r in results]
        descriptions = [r.short_description for r in results]
        modes = [r.player_mode for r in results]
        sources = [r.source for r in results]

        df["genre"] = genres
        df["short_description"] = descriptions
        df["player_mode"] = modes
        df["api_source"] = sources

        save_results(df, CONFIG["OUTPUT_FILE"])

        success_count = sum(1 for r in results if r.success)
        gemini_count = sum(1 for r in results if r.source == "gemini" and r.success)
        deepseek_count = sum(1 for r in results if r.source == "deepseek" and r.success)
        error_count = len(results) - success_count

        print("\n" + "=" * 50)
        print("PROCESSING COMPLETE")
        print("=" * 50)
        print(f"Total games: {len(results)}")
        print(f"Success: {success_count} ({success_count/len(results)*100:.1f}%)")
        print(f"  - Gemini: {gemini_count}")
        print(f"  - DeepSeek (fallback): {deepseek_count}")
        print(f"Failed: {error_count}")
        print(f"Time elapsed: {elapsed:.1f}s")
        print(f"Output: {CONFIG['OUTPUT_FILE']}")

    except ValueError as e:
        logger.error(f"Config error: {e}")
        print(f"\nError: {e}")
        return 1
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\nUnexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())