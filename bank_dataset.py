
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 🔹 1. UCITAVANJE I CISCENJE PODATAKA
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    print(df)
    print(df.columns)
    print(df.isna().sum())
    print(df['balance'].head(50))

    df['job'] = df['job'].fillna('unknown')
    df['marital'] = df['marital'].fillna('unknown')
    df['education'] = df['education'].fillna('unknown')
    df['balance'] = df['balance'].fillna(0)
    df['duration'] = df['duration'].fillna(0)
    df['pdays'] = df['pdays'].fillna(0)
    df['y'] = df['y'].replace('NaN','unknown')

    print(df.isna().sum())
    print(df.duplicated().sum())
    print(df.drop_duplicates())
    return df

# 🔹 2. ANALIZA PRETPLATE PO ZANIMANJU
def analyze_subscription_by_job(df):
    df['job'] = df['job'].str.strip().str.lower().replace({'admin.': 'administrative'})
    df['job'] = df['job'].str.strip().str.lower()
    job_success = df.groupby('job')['y'].value_counts(normalize=True).unstack().fillna(0)
    job_success = job_success.sort_values('yes', ascending=False)
    print(job_success)

    plt.figure(figsize=(10, 6))
    plt.barh(job_success.index, job_success['yes'], color='skyblue')
    plt.xlabel('Proportion of "yes" responses')
    plt.title('Subscription Rate by Job Title')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

# 🔹 3. ZANIMANJE VS KREDITI
def analyze_loans_by_job(df):

    df['job'] = df['job'].str.strip().str.lower().replace({'admin.': 'administrative'})

    job_vs_loan = df.groupby('job')['loan'].value_counts(normalize=True).unstack().fillna(0)
    job_vs_loan = job_vs_loan.sort_values('yes', ascending=False)
    print(job_vs_loan)

    plt.figure(figsize=(10, 6))
    plt.barh(job_vs_loan.index, job_vs_loan['yes'], color='purple')
    plt.xlabel('Have loan')
    plt.title('Personal loan Rate by Job Title')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    job_vs_housing = df.groupby('job')['housing'].value_counts(normalize=True).unstack().fillna(0)
    job_vs_housing = job_vs_housing.sort_values('yes', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(job_vs_housing.index, job_vs_housing['yes'], color='green')
    plt.xlabel('Have housing loan')
    plt.title('Housing Loan Rate by Job Title')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Kombinovano
    jobs_sorted = job_vs_housing.sort_values('yes', ascending=False).index
    loan_rates = job_vs_loan.loc[jobs_sorted, 'yes']
    housing_rates = job_vs_housing.loc[jobs_sorted, 'yes']
    x = np.arange(len(jobs_sorted))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, housing_rates, width, label='Housing Loan', color='green')
    plt.bar(x + width/2, loan_rates, width, label='Personal Loan', color='purple')
    plt.xticks(x, jobs_sorted, rotation=45, ha='right')
    plt.ylabel('Proportion')
    plt.title('Housing vs Personal Loan Rate by Job Title')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

# 🔹 4. GODINE VS KREDITI
def analyze_loans_by_age(df):
    df = df[(df['age'] >= 18) & (df['age'] <= 100)]
    age_loan = df.groupby('age')['loan'].value_counts(normalize=True).unstack().fillna(0)
    age_housing = df.groupby('age')['housing'].value_counts(normalize=True).unstack().fillna(0)

    age_sorted = sorted(set(age_loan.index).intersection(age_housing.index))
    loan_rates = age_loan.loc[age_sorted, 'yes']
    housing_rates = age_housing.loc[age_sorted, 'yes']

    plt.figure(figsize=(12, 6))
    plt.plot(age_sorted, housing_rates, label='Housing Loan', color='green')
    plt.plot(age_sorted, loan_rates, label='Personal Loan', color='purple')
    plt.xlabel('Age')
    plt.ylabel('Proportion with Loan')
    plt.title('Loan Rates by Age')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

# 🔹 5. PRETPLATA PO MESECU
def analyze_monthly_subscription(df):
    month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    monthly_success = df.groupby('month')['y'].value_counts(normalize=True).unstack().fillna(0)
    monthly_success = monthly_success.loc[month_order]

    plt.figure(figsize=(10, 6))
    plt.bar(monthly_success.index, monthly_success['yes'], color='orange')
    plt.title('Subscription Success Rate by Month')
    plt.xlabel('Month')
    plt.ylabel('Subscription Rate (yes)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

# 🔹 6. MARITAL STATUS, EDUCATION VS PRETPLATA
def analyze_marital_and_education(df):
    df['marital'] = df['marital'].str.strip().str.lower().replace({'div.': 'divorced'})
    df['education'] = df['education'].str.strip().str.lower().replace({'sec.': 'secondary', 'unk': 'unknown'})

    marital_success = df.groupby('marital')['y'].value_counts(normalize=True).unstack().fillna(0)
    plt.figure(figsize=(6, 4))
    plt.bar(marital_success.index, marital_success['yes'], color='salmon')
    plt.title('Subscription Rate by Marital Status')
    plt.ylabel('Proportion of "yes"')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    education_success = df.groupby('education')['y'].value_counts(normalize=True).unstack().fillna(0)
    plt.figure(figsize=(8, 5))
    plt.bar(education_success.index, education_success['yes'], color='mediumseagreen')
    plt.title('Subscription Rate by Education Level')
    plt.ylabel('Proportion of "yes"')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

# 🔹 7. PROFIL KLIJENTA KOJI KAZE 'YES'
def profile_yes_customers(df):
    yes_df = df[df['y'] == 'yes']
    print("Average age:", yes_df['age'].mean())
    print("Most common job:", yes_df['job'].value_counts().idxmax())
    print("Most common education:", yes_df['education'].value_counts().idxmax())
    print("Average balance:", yes_df['balance'].mean())
    print("Housing loan (%):", yes_df['housing'].value_counts(normalize=True)['yes'] * 100)
    print("Personal loan (%):", yes_df['loan'].value_counts(normalize=True)['yes'] * 100)

# 🔹 8. KORELACIJA
def correlation_analysis(df):
    numerical_cols = df[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']]
    correlation_matrix = numerical_cols.corr()

    correlation_text = """
    **Correlation:**

    1. Age & Balance: 0.08
       Vrlo slaba pozitivna veza – stariji klijenti imaju blago veci balans.

    2. Age & Duration: -0.00
       Starost ne utice na trajanje poslednjeg kontakta.

    3. Age & Campaign: 0.00
       Godine ne uticu na broj puta koliko je klijent kontaktiran.

    4. Age & Pdays: -0.02
       Nema znacajne veze izmedju starosti i broja dana od poslednjeg kontakta.

    5. Age & Previous: 0.00
       Starost nije povezana sa brojem prethodnih kontakata.

    6. Balance & Duration: 0.02
       Gotovo nepostojeca veza – balans ne utice na trajanje poziva.

    7. Balance & Campaign: -0.01
       Korisnici s visim balansom ne dobijaju znacajno vise ili manje poziva.

    8. Balance & Pdays: 0.00
       Stanje na racunu ne zavisi od broja dana od poslednjeg kontakta.

    9. Balance & Previous: 0.01
       Minimalna pozitivna veza – gotovo zanemarljiva.

    10. Duration & Campaign: -0.08
        Više pokusaja kontakta povezano je sa kracim razgovorima.

    11. Duration & Pdays: -0.00
        Nema veze izmedju trajanja razgovora i broja dana od prethodnog poziva.

    12. Duration & Previous: 0.00
        Duzina razgovora nije povezana s brojem ranijih kontakata.

    13. Campaign & Pdays: -0.09
        Slaba negativna veza – vise kontakata kod klijenata koji nisu skoro kontaktirani.

    14. Campaign & Previous: -0.03
        Vrlo slaba negativna korelacija – zanemarljivo.

    15. Pdays & Previous: 0.45
        Srednje jaka pozitivna korelacija – klijenti kontaktirani pre manje dana cesto su vec bili kontaktirani ranije.
    """

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5])
    ax0 = fig.add_subplot(gs[0, 0])
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax0)
    ax0.set_title("Correlation between numeric columns")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.axis("off")
    ax1.text(0, 1, correlation_text, fontsize=7, va='top', ha='left', wrap=True, family='monospace')
    plt.tight_layout()

# 🔹 9. OUTLIERI I UTICAJ NA PRETPLATU

def outlier_analysis(df):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 1. Boxplot - Outlier detection per variable
    num_cols = ['age', 'balance', 'duration']
    melted_df = df[num_cols].melt(var_name='Variable', value_name='Value')

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Variable', y='Value', data=melted_df)
    plt.title('Boxplot - Outlier Detection per Variable')
    plt.ylabel('Value')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 3. IQR bounds for balance
    Q1 = df['balance'].quantile(0.25)
    Q3 = df['balance'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df['balance'] < lower_bound) | (df['balance'] > upper_bound)]
    outlier_count = len(outliers)

    text_info = f"""
Outlier Analysis - 'Balance'

- Broj outliera: {outlier_count}
- Donja granica: < {lower_bound:.2f}
- Gornja granica: > {upper_bound:.2f}

Odstupanja:
• Negativni outlieri ukazuju na klijente sa velikim dugovanjima.
• Pozitivni outlieri predstavljaju klijente sa vrlo visokim stednim racunima.
"""

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    sns.boxplot(x=df['balance'], color='lightblue', ax=ax0)
    ax0.axvline(lower_bound, color='red', linestyle='--', label=f'Lower bound ({lower_bound:.2f})')
    ax0.axvline(upper_bound, color='green', linestyle='--', label=f'Upper bound ({upper_bound:.2f})')
    ax0.set_title("Boxplot - Account Balance with Outlier Bounds")
    ax0.set_xlabel("Balance")
    ax0.legend()
    ax0.grid(axis='x', linestyle='--', alpha=0.5)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.axis("off")
    ax1.text(0, 1, text_info, fontsize=11, va='top', ha='left', wrap=True, family='monospace')

    plt.tight_layout()

    # 4. Pretplata - Outliers vs Non-Outliers
    outlier_mask = (df['balance'] < lower_bound) | (df['balance'] > upper_bound)
    non_outlier_mask = ~outlier_mask

    yes_outliers = df[outlier_mask]['y'].value_counts(normalize=True).get('yes', 0) * 100
    yes_non_outliers = df[non_outlier_mask]['y'].value_counts(normalize=True).get('yes', 0) * 100

    labels = ['Outliers', 'Non-Outliers']
    yes_rates = [yes_outliers, yes_non_outliers]

    text_info = f"""
Pretplata i 'Balance' Outlieri

• Klijenti sa ekstremnim saldom (outlieri): → {yes_outliers:.2f}%
• Klijenti bez ekstremnog salda: → {yes_non_outliers:.2f}%

Zakljuclak:
Ljudi sa vrlo visokim ili vrlo niskim saldom cesce pristaju
na bankarsku ponudu, verovatno jer imaju specificne finansijske potrebe.
"""

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

    ax0 = fig.add_subplot(gs[0, 0])
    bars = ax0.bar(labels, yes_rates, color=['orange', 'steelblue'])
    ax0.set_ylim(0, max(yes_rates) + 5)
    ax0.set_ylabel('% Yes')
    ax0.set_title('Pretplata po Balance Kategoriji')
    ax0.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax0.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{height:.1f}%', ha='center', va='bottom')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.axis("off")
    ax1.text(0, 1, text_info, fontsize=11, va='top', ha='left', wrap=True, family='monospace')

    plt.tight_layout()


df = load_and_clean_data("bank_dataset.csv")
analyze_subscription_by_job(df)
analyze_loans_by_job(df)
analyze_loans_by_age(df)
analyze_monthly_subscription(df)
analyze_marital_and_education(df)
profile_yes_customers(df)
correlation_analysis(df)
outlier_analysis(df)
plt.show()

df.to_csv('bank_dataset_processed.csv',index=False)