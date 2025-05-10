import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('dirty_financial_transactions.csv')
df = pd.DataFrame(data)
print(df)
print(df.columns)

df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce')

print(df.isna().sum())
df['Transaction_ID'] = df['Transaction_ID'].fillna('unknown')
df['Customer_ID'] = df['Customer_ID'].fillna(0)
df['Quantity'] = df['Quantity'].fillna(0)
df['Price'] = df['Price'].fillna(0)
df['Transaction_Status'] = df['Transaction_Status'].fillna('unknown')
df['Transaction_Date'] = df['Transaction_Date'].ffill()
print(df.isna().sum())


print(df.duplicated().sum())
duplicates = df.drop_duplicates()
print(duplicates)


negative_value = df['Quantity'] < 0
print(negative_value)
print(df['Quantity'])

date_format = pd.to_datetime(df['Transaction_Date'], errors='coerce') <= pd.Timestamp('2025-01-01')
print(date_format)

future_dates = pd.to_datetime(df['Transaction_Date'], errors='coerce') > pd.Timestamp.today()
print(future_dates.sum())


df['Transaction_Status'] = df['Transaction_Status'].str.lower()
success_trans = df['Transaction_Status'] == 'completed'
print(success_trans,df['Transaction_Status'])
print(df['Transaction_Status'].unique())
df['Transaction_Status'] = df['Transaction_Status'].replace({
    'complete': 'completed',
    'in progress': 'pending',
    'fail': 'failed'
})
print(df['Transaction_Status'].unique())
print(df['Transaction_Status'].value_counts())


print(df['Payment_Method'].unique())
df['Payment_Method'] = df['Payment_Method'].str.strip().str.lower()
df['Payment_Method'] = df['Payment_Method'].replace({
    'pay pal': 'PayPal',
    'PayPal': 'PayPal',
    'creditcard': 'Credit Card',
    'credit card': 'Credit Card',
    'paypal': 'PayPal',
})
print(df['Payment_Method'].unique())


print(df['Product_Name'].unique())
df['Product_Name'] = df['Product_Name'].str.strip().str.lower()
df['Product_Name'] = df['Product_Name'].replace({
    # Coffee
    'cof': 'coffee',
    'coffee ': 'coffee',
    'coffee mac': 'coffee machine',
    'coffee ma': 'coffee machine',
    'coffee machin': 'coffee machine',
    'coffee mach': 'coffee machine',
    'coffee machi': 'coffee machine',
    'coffe': 'coffee',
    'co': 'coffee',
    'coffee machine': 'coffee machine',

    # Smartphone
    'smar': 'smartphone',
    'smartp': 'smartphone',
    'smartph': 'smartphone',
    'smartpho': 'smartphone',
    'smartphon': 'smartphone',
    'sma': 'smartphone',
    'smart': 'smartphone',
    'sm': 'smartphone',
    's': 'smartphone',
    'smartphone': 'smartphone',

    # Tablet
    'tabl': 'tablet',
    'tab': 'tablet',
    't': 'tablet',
    'ta': 'tablet',
    'table': 'tablet',

    # Laptop
    'lapt': 'laptop',
    'lapto': 'laptop',
    'lap': 'laptop',
    'la': 'laptop',
    'l': 'laptop',
    'laptop': 'laptop',

    # Headphones
    'headp': 'headphones',
    'headphon': 'headphones',
    'headpho': 'headphones',
    'headph': 'headphones',
    'head': 'headphones',
    'hea': 'headphones',
    'he': 'headphones',
    'h': 'headphones',
    'headphone': 'headphones',
    'headphones': 'headphones',


    'coffee m': 'coffee machine',
    'coff': 'coffee',
    'c': 'coffee',


})
df['Product_Name'] = df['Product_Name'].str.title()
print(df['Product_Name'].unique())


print(df['Price'].unique())
df['Price'] = df['Price'].replace({r'[\$,]': ''}, regex=True).astype(float)
print(df['Price'].head())
print(df['Price'].unique())
negative_prices = df[df['Price'] < 0]
print(negative_prices)
df['Price'] = df['Price'].apply(lambda x: x if x >= 0 else pd.NA)
df['Price'] = pd.to_numeric(df['Price'])
df['Price'] = df['Price'].fillna(0).round(2)
print(df['Price'].unique())


print(df['Quantity'].unique())
df['Quantity'] = df['Quantity'].apply(lambda x: x if x >= 0 else 0)
print(df['Quantity'].unique())
print(df['Quantity'].describe())


print(df['Customer_ID'].unique())
print(df['Customer_ID'].describe())


# Ukupan iznos transakcije

df['Total_Amount'] = df['Quantity'] * df['Price']
print(df['Total_Amount'])

print(df['Total_Amount'].unique())
print(df['Total_Amount'].describe())
product_price_stats = df.groupby('Product_Name')['Total_Amount'].describe()
print(product_price_stats)

highest_values = df[df['Total_Amount'] == df['Total_Amount'].max()]
print(highest_values[['Product_Name', 'Total_Amount']])


plt.figure(figsize=(12, 6))
sns.boxplot(x='Product_Name', y='Total_Amount', data=df)
plt.xticks(rotation=90)
plt.title('Distribucija Ukupnih Iznosa po Proizvodima')
# plt.show()


# Korelacija
correlation_matrix = df[['Quantity', 'Price', 'Total_Amount']].corr()

# Kreiraj figure sa 1 redom i 2 kolone (grafikon levo, tekst desno)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot heatmap sa leva
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, linewidths=0.5, ax=ax[0])

# Naslov za heatmap
ax[0].set_title('Korelacija između Quantity, Price i Total_Amount')

# Dodaj tekst sa desne strane
text = """
Korelacija izmedju varijabli:

1. Quantity i Price:

   -Korelacija od 0.000592 je veoma blizu nule,
    sto znaci da izmedju kolicine proizvoda i njegove cene
    nema gotovo nikakve veze.

   -Povecanje kolicine proizvoda
    ne utice znacajno na cenu proizvoda.

2. Quantity i Total_Amount:

   -Korelacija od 0.46 znaci da
    kolicina ima umeren uticaj na ukupan iznos transakcije.

   -Vise cene proizvoda dovode do vecih ukupnih iznosa,
    ali ovaj odnos nije savrseno jak

3. Price i Total_Amount:

   -korelacija od 0.46 pokazuje da cena proizvoda
    ima umeren uticaj na ukupan iznos.

   -Vece cene dovode do veceg iznosa transakcije,
    sto je ocekivano.
"""

ax[1].text(0.1, 0.5, text, fontsize=10, wrap=True, va='center', ha='left', color='black')
ax[1].axis('off')
plt.subplots_adjust(wspace=0.4)
# plt.show()




# Korelacija izmedju outliera i transakcijskih statusa

q1 = df['Total_Amount'].quantile(0.25)
q3 = df['Total_Amount'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers_total_amount = df[(df['Total_Amount'] < lower_bound) | (df['Total_Amount'] > upper_bound)]

# Korelacija outliera i transakcijskog statusa
outliers_with_status = outliers_total_amount.groupby('Transaction_Status').size()
print(outliers_with_status)

# Boxplot za transakcijski status
plt.figure(figsize=(10, 6))
sns.boxplot(x='Transaction_Status', y='Total_Amount', data=df)
plt.title('Distribucija Total_Amount po Transakcijskom Statusu')
# plt.show()


# Povezanost outliera sa metodom placanja

q1_price = df['Price'].quantile(0.25)
q3_price = df['Price'].quantile(0.75)
iqr_price = q3_price - q1_price
lower_bound_price = q1_price - 1.5 * iqr_price
upper_bound_price = q3_price + 1.5 * iqr_price

outliers_price = df[(df['Price'] < lower_bound_price) | (df['Price'] > upper_bound_price)]

# Korelacija outliera i metode placanja
outliers_with_payment_method = outliers_price.groupby('Payment_Method').size()
print(outliers_with_payment_method)

# Boxplot za metodu placanja
plt.figure(figsize=(10, 6))
sns.boxplot(x='Payment_Method', y='Price', data=df)
plt.title('Distribucija Price po Metodi Plaćanja')
plt.xticks(rotation=45)


# Sezonalnost i outlieri (Povezanost outliera sa vremenskim periodima)

df['Transaction_Month'] = pd.to_datetime(df['Transaction_Date']).dt.month

# Detekcija outliera u ukupnom iznosu
outliers_by_month = df[(df['Total_Amount'] < lower_bound) | (df['Total_Amount'] > upper_bound)].groupby('Transaction_Month').size()
print(outliers_by_month)

# Boxplot za mesece
plt.figure(figsize=(12, 6))
sns.boxplot(x='Transaction_Month', y='Total_Amount', data=df)
plt.title('Distribucija Ukupnih Iznosa po Mesecima')
plt.xticks(rotation=45)
# plt.show()


# Povezivanje outliera sa vrstama proizvoda

outliers_by_product = df[(df['Total_Amount'] < lower_bound) | (df['Total_Amount'] > upper_bound)].groupby('Product_Name').size()
print(outliers_by_product)

# Boxplot za vrste proizvoda
plt.figure(figsize=(12, 6))
sns.boxplot(x='Product_Name', y='Total_Amount', data=df)
plt.title('Distribucija Ukupnih Iznosa po Proizvodima')
plt.xticks(rotation=90)
# plt.show()

from scipy.stats import zscore

df['z_score'] = zscore(df['Total_Amount'])
outliers_zscore = df[df['z_score'].abs() > 3]
print(outliers_zscore)

Q1 = df['Total_Amount'].quantile(0.25)
Q3 = df['Total_Amount'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_iqr = df[(df['Total_Amount'] < lower_bound) | (df['Total_Amount'] > upper_bound)]
print(outliers_iqr)

outliers_by_month = df[df['z_score'].abs() > 3].groupby('Transaction_Month').size()
print(outliers_by_month)
plt.figure(figsize=(10, 6))
df['Transaction_Month'].value_counts().sort_index().plot(kind='bar', color='lightblue')
plt.title('Broj Transakcija po Mesecima')
plt.xlabel('Mesec')
plt.ylabel('Broj Transakcija')
plt.xticks(rotation=0)
# plt.show()


january_transactions = df[df['Transaction_Month'] == 1]
print(january_transactions['Product_Name'].value_counts())


df.to_csv('dirty_financial_transactions_processed.csv')
