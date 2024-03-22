import pandas as pd
import re
def main() -> None:

    df = pd.read_csv("data/rotten_tomatoes_critic_reviews.csv")
    print(df)
    df['review_content'] = df['review_content'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))
    print(df)


if __name__ == '__main__':
    main()
