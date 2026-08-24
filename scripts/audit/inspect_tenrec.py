from pathlib import Path

video_path = Path(r"D:\datasets\TenRec\QK-video.csv")
article_path = Path(r"D:\datasets\TenRec\QK-article.csv")

print("VIDEO EXISTS:", video_path.exists())
print("ARTICLE EXISTS:", article_path.exists())

print("\n===== QK-video first 5 lines =====")
with open(video_path, "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().rstrip())

print("\n===== QK-article first 5 lines =====")
with open(article_path, "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().rstrip())