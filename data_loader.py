"""
data_loader.py — Dataset loading, generation, and preprocessing for Fake News Detection.

Supports:
  1. Auto-generated balanced sample dataset (works out of the box, no download needed)
  2. CSV datasets with 'text' and 'label' columns (e.g., Kaggle Fake News dataset)
  3. LIAR dataset (if downloaded from https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
"""

import os
import re
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalize news text: lowercase, strip HTML tags, remove special characters.
    Keeps sentence structure intact for TF-IDF.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)           # strip HTML
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # remove URLs
    text = re.sub(r"[^a-z\s']", " ", text)          # keep letters + apostrophes
    text = re.sub(r"\s+", " ", text).strip()         # collapse whitespace
    return text


# ---------------------------------------------------------------------------
# Built-in sample dataset (no external download required)
# ---------------------------------------------------------------------------

SAMPLE_REAL = [
    "Scientists at NASA confirmed that the James Webb Space Telescope has captured the deepest infrared image of the universe ever taken, revealing galaxies formed just 600 million years after the Big Bang.",
    "The Federal Reserve raised interest rates by 25 basis points on Wednesday in its continued effort to combat persistent inflation, bringing the federal funds rate to its highest level in over 15 years.",
    "A new peer-reviewed study published in The Lancet found that regular moderate exercise reduces the risk of cardiovascular disease by up to 35 percent in adults over 50.",
    "World leaders gathered at the United Nations General Assembly this week to discuss climate policy, with several nations announcing new commitments to reduce carbon emissions by 2030.",
    "Apple reported quarterly earnings of $90.1 billion in revenue, beating analyst expectations as iPhone sales showed resilience despite a challenging macroeconomic environment.",
    "Researchers at MIT have developed a new battery technology that could store renewable energy twice as efficiently as lithium-ion batteries, potentially transforming the clean energy sector.",
    "The Supreme Court issued a unanimous ruling affirming the right of states to set their own minimum wage laws, citing the Tenth Amendment and precedents from the past three decades.",
    "Global temperatures in 2023 were the highest on record, according to data released by the World Meteorological Organization, continuing a trend of warming observed since the industrial revolution.",
    "A major earthquake measuring 6.8 on the Richter scale struck off the coast of Japan early Tuesday morning. Authorities issued a brief tsunami warning that was later lifted after no significant waves were detected.",
    "The European Union reached a landmark agreement on artificial intelligence regulation, establishing strict rules for high-risk AI applications in healthcare, law enforcement, and financial services.",
    "Brazil's Amazon deforestation fell by 50 percent in the first half of the year compared to the same period last year, according to official government satellite monitoring data.",
    "Pfizer's updated COVID-19 vaccine showed strong immune response against the latest circulating variants in Phase 3 clinical trials, according to results published in the New England Journal of Medicine.",
    "The International Monetary Fund revised its global growth forecast upward to 3.1 percent for the coming year, citing resilient consumer spending in the United States and recovery in emerging markets.",
    "Engineers at SpaceX successfully launched and recovered the Falcon 9 booster for a record 20th time, demonstrating the rocket's reliability and pushing down launch costs significantly.",
    "A new archaeological dig in southern Turkey has uncovered a 10,000-year-old settlement with evidence of early agricultural practices, rewriting the timeline of human civilization in the region.",
    "The World Health Organization declared the end of the mpox public health emergency of international concern, noting a significant decline in reported cases across all previously affected regions.",
    "Congress passed a bipartisan infrastructure bill allocating $65 billion for broadband expansion, aiming to bring high-speed internet access to rural communities by the end of the decade.",
    "Astronomers confirmed the discovery of an exoplanet in the habitable zone of a nearby star system, roughly 1.4 times the size of Earth, with atmospheric conditions that could support liquid water.",
    "Japan's economy grew by 1.9 percent in the second quarter, outperforming expectations, driven largely by strong export performance and a recovery in domestic tourism after the pandemic.",
    "The United States and the European Union signed a new data privacy framework designed to replace the previously invalidated Privacy Shield agreement, ensuring legal cross-border data transfers.",
    "A clinical trial led by Johns Hopkins University demonstrated that a combination therapy approach reduced drug-resistant tuberculosis treatment duration from 18 months to just 6 months.",
    "Renewable energy sources accounted for over 30 percent of global electricity generation last year for the first time, led by rapid expansion of solar capacity in China, India, and the United States.",
    "The Bank of England held interest rates steady for the third consecutive meeting, signaling that the central bank believes inflation is moving sustainably toward its 2 percent target.",
    "Doctors at the Cleveland Clinic performed the world's first whole-eye transplant, giving a patient who lost his eye in an electrical accident the chance of partial sight restoration.",
    "A federal jury found the defendant guilty on all 34 counts of falsifying business records in a case that drew significant media attention and public debate about accountability and the rule of law.",
]

SAMPLE_FAKE = [
    "BREAKING: Government secretly adding mind-control chemicals to tap water to make citizens docile and obedient. Scientists who tried to expose this have mysteriously disappeared.",
    "EXCLUSIVE: The moon landing was entirely staged in a Hollywood studio. Newly leaked documents from a NASA whistleblower prove that Stanley Kubrick directed the fake footage.",
    "SHOCKING TRUTH: Doctors and pharmaceutical companies are hiding the 100% natural cancer cure that has been suppressed for decades to keep patients paying for expensive treatments.",
    "ALERT: 5G towers are being used to beam radiation that activates tracking nanobots injected through COVID vaccines. Thousands of citizens are already under surveillance.",
    "EXPOSED: Chemtrails from commercial aircraft contain biological agents designed to reduce human fertility as part of a globalist depopulation agenda approved at secret Davos meetings.",
    "CONFIRMED: A secret underground city beneath Denver Airport houses thousands of elite families who are prepared to survive a planned extinction-level event scheduled for next year.",
    "URGENT: The Federal Reserve is printing unlimited cash and shipping it overseas in unmarked planes to fund a shadow government that controls all world leaders through blackmail.",
    "REVEALED: Birds are not real. They are federally deployed surveillance drones that charge on power lines and report citizen activities to a centralized AI monitoring system.",
    "DEVELOPING: Scientists have discovered a free energy device that eliminates the need for oil and electricity, but oil companies paid the patent office to bury it permanently.",
    "MASSIVE COVER-UP: Top virologists admit in leaked emails that all respiratory viruses are actually manufactured in labs and released strategically before elections to control populations.",
    "FORBIDDEN KNOWLEDGE: Ancient pyramids are actually giant antennas built by advanced extraterrestrials to communicate with their home galaxy, as proven by newly declassified CIA files.",
    "BREAKING: Hollywood actor reveals that all major celebrities are required to join a secret satanic cult or their careers are destroyed by shadowy agents in the entertainment industry.",
    "EXPOSED: The global warming narrative is a trillion-dollar hoax invented by international bankers to justify a carbon tax that funnels money to offshore accounts controlled by three families.",
    "ALERT: Drinking bleach mixed with lemon juice cures over 200 diseases, but the medical establishment has banned all research because it would destroy the pharmaceutical industry.",
    "CONFIRMED: The real cure for diabetes is a spice found in every kitchen cabinet, but doctors are legally prohibited from revealing this after a secret agreement with insulin manufacturers.",
    "BREAKING NEWS: Scientists at a hidden Antarctic base have made contact with beings from another dimension and the footage has been confiscated by intelligence agencies in all 5 major nations.",
    "URGENT ALERT: The global elite are engineering a fake alien invasion to be used as a pretext to declare world martial law and install a one-world government within the next 18 months.",
    "EXPOSED: Social media companies are reading all private messages and selling your innermost secrets to foreign governments who use them to synthesize psychological control algorithms.",
    "REVEALED: Genetically modified food contains a gene sequence that activates a dormant virus in the human genome, causing sterility in the second generation of those who consume it.",
    "SHOCKING: Hospitals are secretly euthanizing patients who refuse experimental treatments and harvesting their organs to sell to the ultra-wealthy through an underground medical black market.",
    "CONFIRMED: Elon Musk does not exist and is a fictional persona created by a consortium of Silicon Valley investors to hide the true ownership of Tesla, SpaceX, and other tech companies.",
    "BREAKING: Underground tunnels discovered beneath major cities are being used to traffic children by a global network of politicians, celebrities, and corporate executives.",
    "FORBIDDEN: Scientists who studied the real effects of fluoride in water were all suicided to prevent the public from learning it calcifies the pineal gland and blocks spiritual awakening.",
    "EXPOSED: All major historical events of the past century, including world wars, financial crashes, and pandemics, were orchestrated by the same twelve banking families who own all central banks.",
    "ALERT: New legislation being secretly drafted would require all citizens to receive a mandatory brain implant linked to the government cloud to receive healthcare and social services.",
]

def build_sample_dataset() -> pd.DataFrame:
    """
    Build a balanced sample dataset from the curated lists above.
    Returns a DataFrame with 'text' and 'label' columns (0=Real, 1=Fake).
    """
    real_df = pd.DataFrame({
        "text": SAMPLE_REAL,
        "label": [0] * len(SAMPLE_REAL),
    })
    fake_df = pd.DataFrame({
        "text": SAMPLE_FAKE,
        "label": [1] * len(SAMPLE_FAKE),
    })
    df = pd.concat([real_df, fake_df], ignore_index=True).sample(frac=1, random_state=42)
    df["text"] = df["text"].apply(clean_text)
    return df


# ---------------------------------------------------------------------------
# CSV / LIAR dataset loaders
# ---------------------------------------------------------------------------

def load_csv_dataset(path: str) -> pd.DataFrame:
    """
    Load a CSV dataset with at minimum 'text' and 'label' columns.
    Label must be 0 (real) or 1 (fake), OR the string 'REAL'/'FAKE'.
    """
    df = pd.read_csv(path)

    # Normalize label column
    if df["label"].dtype == object:
        df["label"] = df["label"].str.upper().map({"REAL": 0, "FAKE": 1, "TRUE": 0, "FALSE": 1})

    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 20]  # drop near-empty rows
    return df


def load_liar_dataset(data_dir: str) -> pd.DataFrame:
    """
    Load the LIAR dataset from a local directory.
    Expected files: train.tsv, valid.tsv, test.tsv
    Download from: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

    LIAR labels: pants-fire, false, barely-true → Fake (1)
                 half-true, mostly-true, true    → Real (0)
    """
    FAKE_LABELS = {"pants-fire", "false", "barely-true"}
    cols = [
        "id", "label", "statement", "subject", "speaker",
        "job_title", "state_info", "party_affiliation",
        "barely_true_counts", "false_counts", "half_true_counts",
        "mostly_true_counts", "pants_on_fire_counts", "context",
    ]
    dfs = []
    for split in ("train.tsv", "valid.tsv", "test.tsv"):
        fpath = os.path.join(data_dir, split)
        if os.path.exists(fpath):
            part = pd.read_csv(fpath, sep="\t", header=None, names=cols)
            dfs.append(part)

    if not dfs:
        raise FileNotFoundError(f"No LIAR .tsv files found in {data_dir}")

    df = pd.concat(dfs, ignore_index=True)
    df["text"] = df["statement"].apply(clean_text)
    df["label"] = df["label"].apply(lambda x: 1 if str(x).lower() in FAKE_LABELS else 0)
    return df[["text", "label"]].dropna()


# ---------------------------------------------------------------------------
# Master loader
# ---------------------------------------------------------------------------

def load_dataset(source: str = "sample") -> pd.DataFrame:
    """
    Unified entry point for dataset loading.

    Args:
        source: 'sample'      — use built-in balanced dataset (default)
                '<path>.csv'  — load a local CSV file
                '<dir>'       — load LIAR dataset from a directory

    Returns:
        pd.DataFrame with columns: ['text', 'label']
    """
    if source == "sample":
        return build_sample_dataset()
    elif source.endswith(".csv") and os.path.isfile(source):
        return load_csv_dataset(source)
    elif os.path.isdir(source):
        return load_liar_dataset(source)
    else:
        print(f"[data_loader] Source '{source}' not found — falling back to sample dataset.")
        return build_sample_dataset()
