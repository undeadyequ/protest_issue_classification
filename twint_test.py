"""
1. condition:
    keyword:
    time
    limit
2. scrape photos

3. check json file

-> dir
keyword1 -> category1
keyword2 -> category2
...

"""
import twint

import os
import pathlib
import json

def save_img_from_url(image_url, img_f):
    import requests
    img_data = requests.get(image_url).content
    with open(img_f, 'wb') as handler:
        handler.write(img_data)


def condition_generate():
    """

    :return:
        dict{str: dict{str: list}}
    """
    condition = {}
    categories = ["Biden", "Trump", "Clinton", "Obama", "Romney"]
    times = [("2020-10-01", "2020-11-06"),
             ("2020-10-01", "2020-11-06"),
             ("2016-10-01", "2016-11-10"),
             ("2012-10-01", "2012-11-10"),
             ("2012-10-01", "2012-11-10")]
    keywords_format = ["{} campaign rally"]
    limit = [100] * len(categories)

    # keywords
    keywords = []
    for cate in categories:
        keyword_cate = []
        for kf in keywords_format:
            keyword_cate.append(kf.format(cate))
        keywords.append(keyword_cate)

    for i, cate in enumerate(categories):
        condition[cate] = {"keyword": keywords[i],
                           "time": times[i],
                           "limit": limit[i]}
    return condition


def scrape_twitter_img2(search_condition):
    """

    :param search_condition:
            dict{str: dict{str: list}}

    :return:
    """
    # set out.json
    for cat, cond in search_condition.items():
        dir_cat = cat
        for kw in cond["keyword"]:
            kw_dir = kw.replace(" ", "_")
            kw_path = os.path.join(dir_cat, kw_dir)

            outjson = os.path.join(kw_path, "out.json")
            command = "twint -s \"{}\" --images -o {} --json --popular-tweets --limit {} --since {} --until {}".\
                format(kw, outjson, cond["limit"], cond["time"][0], cond["time"][1])
            outjson_n = 0

            # scrape info into out.json
            pathlib.Path(kw_path).mkdir(parents=True, exist_ok=True)
            while outjson_n < cond["limit"] - 1:
                print("start execute command: ", command)
                os.system(command)
                # check number of out.json and re-scrape if not enough
                if os.path.isfile(outjson):
                    with open(outjson, 'r') as handle:
                        json_data = [json.loads(line) for line in handle]
                        outjson_n = len(json_data)

            # download image by reading url written in out.json
            pathlib.Path(kw_path).mkdir(parents=True, exist_ok=True)
            with open(outjson, 'r') as handle:
                json_data = [json.loads(line) for line in handle]
            for i, d in enumerate(json_data):
                for j, url in enumerate(d["photos"]):
                    save_img_from_url(url, os.path.join(kw_path, str(i+j)+".jpg"))


def main():
    # 1. condition
    condition = condition_generate()
    # 2. scrape photos
    # 3. check json file
    scrape_twitter_img2(condition)


if __name__ == '__main__':
    main()