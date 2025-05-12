import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def load_data(file_path):
    data = pd.read_csv('linkdin_Job_data.csv')
    df = pd.DataFrame(data)
    return df

def clean_data(df):
    df = df.apply(lambda x: x.fillna(0) if x.dtype in ['float64', 'int64'] else x.fillna('Unknown'))
    df = df.replace([None, 'ERROR', 'UNKNOWN'], 'Unknown')
    if 'Column1' in df.columns:
        df = df.drop(columns=['Column1'])
    return df

def filter_data(df):
    top_ten_positions = df['job'].value_counts().head(10)
    return top_ten_positions

def job_count_by_location(df):
    location_counts = df['location'].value_counts()
    return location_counts

def distribution_by_type_of_work(df):
    distr_per_work_type = df['work_type'].value_counts()
    return distr_per_work_type

def clean_no_of_application(df):
    df['no_of_application'] = df['no_of_application'].astype(str)
    df['no_of_application'] = df['no_of_application'].str.extract(r'([\d,]+)')
    df['no_of_application'] = df['no_of_application'].str.replace(',', '')
    df['no_of_application'] = pd.to_numeric(df['no_of_application'], errors='coerce').fillna(0).astype(int)

def clean_linkedin_followers(df):
    df['linkedin_followers'] = df['linkedin_followers'].astype(str)
    df['linkedin_followers'] = df['linkedin_followers'].str.extract(r'([\d,]+)')
    df['linkedin_followers'] = df['linkedin_followers'].str.replace(',', '')
    df['linkedin_followers'] = pd.to_numeric(df['linkedin_followers'], errors='coerce').fillna(0).astype(int)
    return df

def average_applications(df):
    clean_no_of_application(df)
    return df['no_of_application'].astype(int).mean()

def correlation_followers_applications(df):
    clean_no_of_application(df)
    clean_linkedin_followers(df)
    df['no_of_application'] = pd.to_numeric(df['no_of_application'], errors='coerce')
    return df['linkedin_followers'].corr(df['no_of_application'])

def remote_job_frequency(df):
    return df['full_time_remote'].value_counts(normalize=True)

def top_hiring_companies(df, n=10):
    return df['company_name'].value_counts().head(n)

def full_vs_part_time(df):
    return df['work_type'].value_counts()

def job_post_age_distribution(df):
    df['posted_day_ago'] = df['posted_day_ago'].astype(str).str.extract(r'(\d+)')
    df['posted_day_ago'] = pd.to_numeric(df['posted_day_ago'], errors='coerce').fillna(0).astype(int)
    return df['posted_day_ago'].value_counts().sort_index()

def average_applications_per_location(df):
    clean_no_of_application(df)
    return df.groupby('location')['no_of_application'].mean().sort_values(ascending=False)

def plot_top_positions(df):
    top_jobs = df['job'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_jobs.values, y=top_jobs.index, palette="Blues_d", hue=top_jobs.index, legend=False)
    plt.title("Top 10 Najtrazenijih Pozicija")
    plt.xlabel("Broj oglasa")
    plt.ylabel("Pozicija")
    plt.tight_layout()
    plt.show()

def plot_avg_applications_by_location(df):
    clean_no_of_application(df)
    avg_apps = df.groupby('location')['no_of_application'].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_apps.values, y=avg_apps.index, palette="mako", hue=avg_apps.index, legend=False)
    plt.title("Prosecan broj aplikacija po lokaciji (Top 10)")
    plt.xlabel("Prosecan broj aplikacija")
    plt.ylabel("Lokacija")
    plt.tight_layout()
    plt.show()

def plot_posted_days_distribution(df):
    df['posted_day_ago'] = df['posted_day_ago'].astype(str).str.extract(r'(\d+)')
    df['posted_day_ago'] = pd.to_numeric(df['posted_day_ago'], errors='coerce').fillna(0).astype(int)
    day_dist = df['posted_day_ago'].value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=day_dist.index, y=day_dist.values, marker='o', color='coral')
    plt.title("Broj oglasa po danima (od objave)")
    plt.xlabel("Broj dana od objave")
    plt.ylabel("Broj oglasa")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_remote_job_ratio(df):
    remote_counts = df['full_time_remote'].value_counts(normalize=True) * 100

    plt.figure(figsize=(6, 4))
    sns.barplot(x=remote_counts.index, y=remote_counts.values, palette=sns.color_palette("coolwarm", len(remote_counts)), hue=remote_counts.index, legend=False)
    plt.title("Udeo Remote Poslova (%)", fontsize=14)
    plt.xlabel("Remote status")
    plt.ylabel("Udeo (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=50, ha='right', fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_corr_heatmap(df):
    clean_no_of_application(df)
    clean_linkedin_followers(df)

    df['no_of_application'] = pd.to_numeric(df['no_of_application'], errors='coerce')
    df['linkedin_followers'] = pd.to_numeric(df['linkedin_followers'], errors='coerce')

    corr_matrix = df[['linkedin_followers', 'no_of_application']].corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f', linewidths=0.5)
    plt.title('Korelacija izmedju broja LinkedIn pratilaca i broja aplikacija', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_boxplot(df, column):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot za {column}")
    plt.show()

    plot_boxplot(df, 'no_of_application')

def main(file_path):
    df = load_data(file_path)
    df = clean_data(df)

    top_ten_positions = filter_data(df)
    location_counts = job_count_by_location(df)
    work_type_distribution = distribution_by_type_of_work(df)
    avg_applications = average_applications(df)
    correlation = correlation_followers_applications(df)
    remote_frequency = remote_job_frequency(df)
    job_post = job_post_age_distribution(df)
    avg_app = average_applications_per_location(df)

    plot_top_positions(df)
    plot_avg_applications_by_location(df)
    plot_posted_days_distribution(df)
    plot_remote_job_ratio(df)
    plot_corr_heatmap(df)
    # plot_boxplot(df, 'no_of_application')

    print("Top 10 najtrazenijih pozicija:\n", top_ten_positions)
    print("\nBroj pozicija po lokaciji:\n", location_counts)
    print("\nDistribucija po tipu zaposlenja:\n", work_type_distribution)
    print("\nProsecan broj aplikacija po poslu:", avg_applications)
    print("\nKorelacija izmedju broja LinkedIn pratilaca i broja aplikacija:", correlation)
    print("\nFrekvencija trazenja remote pozicija:\n", remote_frequency)
    print(job_post)
    print('Average apps per locations:\n', avg_app)


if __name__ == "__main__":
    file_path = 'linkdin_Job_data.csv'
    df = main(file_path)





