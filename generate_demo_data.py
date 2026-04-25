"""
Synthetic Demo Dataset Generator — Job Hiring Dataset
=======================================================
Generates a realistic hiring decision dataset with embedded biases
across race, age group, and education level.
"""

import pandas as pd
import numpy as np


def generate_demo_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic job hiring dataset with intentional biases.

    Sensitive attributes: race, age_group, education
    Target: hired (0/1)

    Biases embedded:
      - Race bias: White/Asian hired more; Black/Hispanic hired less
      - Age bias: 41-60 age group hired less despite experience
      - Proxy: college_tier correlates strongly with race
      - Missing: skill_score missing more for Black/Hispanic applicants
    """
    np.random.seed(random_state)

    # --- Sensitive attributes ---
    race = np.random.choice(
        ["White", "Asian", "Black", "Hispanic", "Other"],
        size=n_samples,
        p=[0.40, 0.20, 0.20, 0.15, 0.05]
    )

    age_group = np.random.choice(
        ["21-30", "31-40", "41-50", "51-60"],
        size=n_samples,
        p=[0.30, 0.35, 0.25, 0.10]
    )

    education = np.random.choice(
        ["HighSchool", "Bachelor", "Master", "PhD"],
        size=n_samples,
        p=[0.20, 0.45, 0.25, 0.10]
    )

    # --- Features ---
    experience_years = np.random.poisson(7, n_samples).clip(0, 35)

    # Older groups have more experience on average
    exp_boost = np.where(age_group == "41-50", 5,
                np.where(age_group == "51-60", 10, 0))
    experience_years = (experience_years + exp_boost).clip(0, 35)

    skill_score = np.random.normal(65, 15, n_samples).clip(10, 100).astype(int)

    # Bias: skill_score slightly lower for Black/Hispanic (biased assessments)
    skill_score[np.isin(race, ["Black", "Hispanic"])] -= 8

    interview_score = np.random.normal(60, 18, n_samples).clip(10, 100).astype(int)

    # Bias: interview score lower for 41-50 and 51-60 (age bias in interviews)
    interview_score[np.isin(age_group, ["41-50", "51-60"])] -= 10

    # Proxy variable: college_tier correlates with race
    # (simulating historically segregated educational access)
    tier_base = {"White": 1, "Asian": 1, "Black": 3, "Hispanic": 3, "Other": 2}
    college_tier = np.array([
        tier_base[r] + np.random.randint(-1, 2) for r in race
    ]).clip(1, 4)

    # Another feature
    num_previous_jobs = np.random.poisson(3, n_samples).clip(0, 10)

    # --- Missing data bias: skill_score missing more for Black/Hispanic ---
    skill_float = skill_score.astype(float)
    # Baseline 5% missing for all
    miss_mask = np.random.random(n_samples) < 0.05
    # Extra 15% missing for Black/Hispanic
    miss_minority = (np.isin(race, ["Black", "Hispanic"]) &
                     (np.random.random(n_samples) < 0.15))
    skill_float[miss_mask | miss_minority] = np.nan

    # --- Target: hired decision (biased) ---
    edu_score = {"HighSchool": 0, "Bachelor": 0.2, "Master": 0.35, "PhD": 0.45}
    base_prob = (
        0.20
        + np.array([edu_score[e] for e in education])
        + 0.20 * (skill_score - 10) / 90
        + 0.15 * (interview_score - 10) / 90
        + 0.10 * (experience_years / 35)
    )

    # Inject bias
    bias = np.zeros(n_samples)
    bias[np.isin(race, ["Black", "Hispanic"])] -= 0.18   # Race bias
    bias[race == "Other"]                       -= 0.08   # Minority bias
    bias[np.isin(age_group, ["41-50", "51-60"])] -= 0.14  # Age bias
    # Intersectional: Black + 41-60 gets hit hardest
    bias[(np.isin(race, ["Black", "Hispanic"])) &
         (np.isin(age_group, ["41-50", "51-60"]))] -= 0.10

    prob = (base_prob + bias).clip(0.05, 0.95)
    hired = (np.random.random(n_samples) < prob).astype(int)

    df = pd.DataFrame({
        "race":              race,
        "age_group":         age_group,
        "education":         education,
        "experience_years":  experience_years,
        "skill_score":       skill_float,
        "interview_score":   interview_score,
        "college_tier":      college_tier,
        "num_previous_jobs": num_previous_jobs,
        "hired":             hired,
    })

    return df


if __name__ == "__main__":
    df = generate_demo_dataset()
    print(f"Generated {len(df)} records")
    print(f"\nHire rates by race:")
    print(df.groupby("race")["hired"].mean().round(3))
    print(f"\nHire rates by age_group:")
    print(df.groupby("age_group")["hired"].mean().round(3))
    print(f"\nHire rates by education:")
    print(df.groupby("education")["hired"].mean().round(3))
    print(f"\nMissing skill_score by race:")
    print(df.groupby("race")["skill_score"].apply(lambda x: x.isnull().mean()).round(3))
    df.to_csv("demo_data.csv", index=False)
    print("\nSaved to demo_data.csv")
