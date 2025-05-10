import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, probplot, spearmanr


def load_data(file_path):
    data = pd.read_csv('dirty_cafe_sales.csv')
    df = pd.DataFrame(data)
    return df


def clean_data(df):
    df = df.apply(lambda x: x.fillna(0) if x.dtype in ['float64', 'int64'] else x.fillna('Unknown'))
    df = df.replace([None, 'ERROR', 'UNKNOWN'], 'Unknown')
    return df


def preprocess_spent(df):
    df['Total Spent'] = df['Total Spent'].replace('Unknown', 0)
    df['Total Spent'] = pd.to_numeric(df['Total Spent'], errors='coerce')
    return df


def transactions_per_location(df):
    return df.groupby('Location')['Transaction ID'].count()


def spending_per_payment_method(df):
    return df.groupby('Payment Method')['Total Spent'].sum()


def top_5_expensive_transactions(df):
    return df['Total Spent'].nlargest(5)


def average_spent_per_payment_method(df):
    return df.groupby('Payment Method')['Total Spent'].mean()


def most_frequent_items(df):
    return df[['Item', 'Location']].groupby('Location')['Item'].value_counts()


def transactions_above_10(df):
    return df[df['Total Spent'] > 10]


def avg_spent_per_transaction_and_location(df):
    """Average spending per transaction and location."""
    return df.groupby(['Location', 'Transaction ID'])['Total Spent'].mean()


def day_with_highest_spending(df):
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
    daily_spending = df.groupby(df['Transaction Date'].dt.date)['Total Spent'].sum()
    day_with_most_spent = daily_spending.idxmax()
    max_spent_value = daily_spending.max()
    return day_with_most_spent, max_spent_value


def most_profitable_item(df):
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
    df['Price Per Unit'] = pd.to_numeric(df['Price Per Unit'], errors='coerce').fillna(0)
    df['Total Profit'] = df['Quantity'] * df['Price Per Unit']
    profit_per_item = df.groupby('Item')['Total Profit'].sum()
    top_item = profit_per_item.idxmax()
    top_profit = profit_per_item.max()
    return top_item, top_profit


def most_common_payment_method(df):
    return df['Payment Method'].mode()[0]


def transactions_below_average(df):
    average_spent = df['Total Spent'].mean()
    below_average = df[df['Total Spent'] < average_spent]
    return below_average


def profit_by_quarter(df):
    df['Quarter'] = df['Transaction Date'].dt.to_period('Q')
    quarterly_profit = df.groupby('Quarter')['Total Profit'].sum().reset_index()
    return quarterly_profit


def profit_by_month(df):
    df['Month'] = df['Transaction Date'].dt.to_period('M')
    monthly_profit = df.groupby('Month')['Total Profit'].sum().reset_index()
    return monthly_profit


def plot_quarterly_profit(quarterly_profit):
    plt.figure(figsize=(10, 6))
    plt.plot(quarterly_profit['Quarter'].astype(str), quarterly_profit['Total Profit'], marker='o', linestyle='-', color='b')
    plt.title('Profit per Quarter for 2023')
    plt.xlabel('Quarter')
    plt.ylabel('Profit per Quarter ($)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_monthly_profit(monthly_profit):
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_profit['Month'].astype(str), monthly_profit['Total Profit'], marker='o', linestyle='-', color='g')
    plt.title('Profit per Month for 2023')
    plt.xlabel('Month')
    plt.ylabel('Profit per Month ($)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def correlation_analysis(df):

    numerical_cols = df[['Total Spent', 'Quantity', 'Price Per Unit', 'Total Profit']]

    # Compute correlation matrix
    correlation_matrix = numerical_cols.corr()

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.2)
    plt.title('Correlation between different variables')

    correlation_text = """
    **Korelacija:**

    1. Total Spent & Quantity: 0.58
       Vise potrosnje cesto znaci i vise kupljenih jedinica.

    2. Total Spent & Price Per Unit: 0.51
       Korisnici koji trose vise cesto biraju skuplje proizvode.

    3. Total Spent & Total Profit: 0.79
       Visa potrosnja korisnika vodi ka vecem profitu firme.

    4. Quantity & Price Per Unit: 0.00
       Nema veze izmedju kolicine i cene proizvoda.

    5. Quantity & Total Profit: 0.68
       Vise prodatih komada povecava ukupni profit.

    6. Price Per Unit & Total Profit: 0.65
       Skuplji proizvodi doprinose vecem profitu.
    """

    # Kreiramo layout sa 1 redom i 2 kolone
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

    # Prva kolona: heatmap
    ax0 = fig.add_subplot(gs[0, 0])
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax0)
    ax0.set_title("Korelacija izmedju varijabli")

    # Druga kolona: tekst
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.axis("off")
    ax1.text(0, 1, correlation_text, fontsize=12, va='top', ha='left', wrap=True)

    plt.tight_layout()
    plt.show()


def test_normality(df, column):
    """Test normalnosti koristeći Shapiro-Wilk test i vizualizaciju distribucije"""

    # Shapiro-Wilk test
    stat, p_value = shapiro(df[column].dropna())
    print(f"Shapiro-Wilk Test za {column}: Statistika = {stat}, P-vrednost = {p_value}")

    # Testiranje normalnosti na osnovu P-vrednosti
    if p_value > 0.05:
        print(f"{column} have normal distribution.")
    else:
        print(f"{column} not normal distribution.")


    plt.figure(figsize=(12, 6))

    # 1. Histogram sa KDE (Kernel Density Estimate)
    plt.subplot(1, 2, 1)
    sns.histplot(df[column].dropna(), kde=True, bins=30)
    plt.title(f'Distribution of {column} with KDE')
    plt.xlabel(column)
    plt.ylabel('Frequency')

    # 2. Q-Q Plot (Quantile-Quantile Plot)
    plt.subplot(1, 2, 2)
    probplot(df[column].dropna(), dist="norm", plot=plt)
    plt.title(f'Q-Q Plot for {column}')
    plt.tight_layout()
    plt.show()


def main(file_path):
    df = load_data(file_path)
    df = clean_data(df)
    df = preprocess_spent(df)

    # 1. Transactions per location
    print(transactions_per_location(df))

    # 2. Total spending by payment method
    print(spending_per_payment_method(df))

    # 3. Top 5 most expensive transactions
    print(top_5_expensive_transactions(df))

    # 4. Average spending by payment method
    print(average_spent_per_payment_method(df))

    # 5. Most frequent items per location
    print(most_frequent_items(df))

    # 6. Transactions above 10
    print(transactions_above_10(df))

    # 7. Average spent per transaction and location
    print(avg_spent_per_transaction_and_location(df))

    # 8. Day with the highest spending
    day, max_spent = day_with_highest_spending(df)
    print(f"Day with highest spending: {day}, Amount: {max_spent}")

    # 9. Most profitable item
    item, profit = most_profitable_item(df)
    print(f"Most profitable item: {item}, Profit: {profit}")

    # 10. Most common payment method
    print(most_common_payment_method(df))

    # 11. Transactions below average
    print(transactions_below_average(df))

    # 12. Profit by quarter
    quarterly_profit = profit_by_quarter(df)
    plot_quarterly_profit(quarterly_profit)

    # 13. Profit by month
    monthly_profit = profit_by_month(df)
    plot_monthly_profit(monthly_profit)


    df.to_csv('dirty_cafe_sales_processed.csv', index=False)



if __name__ == "__main__":
    file_path = 'dirty_cafe_sales.csv'
    main(file_path)

    df = load_data(file_path)
    df = clean_data(df)
    df = preprocess_spent(df)



    test_normality(df, 'Total Spent')

    test_normality(df, 'Quantity')
    test_normality(df, 'Price Per Unit')
    test_normality(df, 'Total Profit')

    correlation_analysis(df)

    df.to_csv('dirty_cafe_sales.csv', index=False)



# 1. Test normalnosti (Shapiro-Wilk test):

# Total Spent, Quantity, Price Per Unit, i Total Profit:
# Svi ovi podaci imaju veoma malu P-vrednost (manju od 0.05),
# što znasi da svi ovi podaci nemaju normalnu distribuciju.
# To je ocekivano za mnoge poslovne i ekonomske podatke,
# koji cesto prate raspodelu koja nije normalna (npr. pozitivno skewed distribucije).
#


# Napomena: Takodje, upozorenje "For N > 5000,
# znaci da za velike uzorke
# (kao što je 10.000 redova), Shapiro-Wilk test mozda nece biti potpuno precizan.
# U tom slucaju, za velike uzorke, koristiti druge testove normalnosti,
# poput Anderson-Darling ili Kolmogorov-Smirnov sa N > 5000.
#


# 2. Spearmanov test korelacije:

# Total Spent i Quantity: Statistika = 0.59, P-vrednost = 0.0.
# Postoji statisticki znacajna korelacija, sto znaci da veca potrosnja obicno znaci
# i vecu kolicinu kupljenih proizvoda.
#

# Total Spent i Price Per Unit: Statistika = 0.52, P-vrednost = 0.0.
# Takodje postoji statisticki znacajna korelacija,
# sto sugerise da korisnici koji trose vise cesto biraju skuplje proizvode.
#

# Total Spent i Total Profit: Statistika = 0.73, P-vrednost = 0.0.
# Veca potrosnja povezana je sa vecim profitom, sto je i ocekivano.
#


# Quantity i Price Per Unit: Statistika = -0.00035, P-vrednost = 0.97.
# Ovdje nema statisticki znacajne korelacije,
# sto znaci da kolicina kupljenih proizvoda nije povezana sa cenom po jedinici proizvoda.
#


# Quantity i Total Profit: Statistika = 0.68, P-vrednost = 0.0.
# Postoji statisticki znacajna korelacija izmedju kolicine kupljenih proizvoda i ukupnog profita.
#


# Price Per Unit i Total Profit: Statistika = 0.65, P-vrednost = 0.0.
# Statisticki znacajna korelacija, sto ukazuje na to da skuplji proizvodi obicno generisu veci profit.
#
