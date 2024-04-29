import csv

import pymysql

def db_connect(database):
    db_con = pymysql.connect(
        host="dev.danawa.com",
        port=3306,
        user='DEdevelop1_E',
        password="goqkfkrl^(**",
        db=database,
        charset='utf8'
    )
    db_con.query("SET NAMES utf8mb4")
    db_con.commit()
    cursor = db_con.cursor()
    return db_con, cursor


def return_sql(key):
    sql_dict = {
        # 전체 컬럼
        "all_column_sql": "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'dbBoard'",
        # 카테고리 매칭된 뉴스 개수(중복 허용)
        "duplicate_category_news_count_sql" : '''
            SELECT COUNT(*)
            FROM tNormalList L
            INNER JOIN tNormalContent C ON L.nListSeq = C.nListSeq
            INNER JOIN tNormalListCateProduct CP ON (L.nListSeq = CP.nListSeq AND CP.nCategorySeq1 <> 0)
            WHERE L.nBoardSeq IN (60,61,62,63,64,65,66,67,68,130);
            ''',
        # 카테고리 매칭된 뉴스 데이터 샘플 조회(중복 허용)
        "duplicate_category_news_sql": '''
            SELECT L.nListSeq, L.nBoardSeq, L.sTitle, C.sContent, CP.nCategorySeq1, CP.nCategorySeq2, CP.nCategorySeq3, CP.nCategorySeq4
            FROM tNormalList L
            INNER JOIN tNormalContent C ON L.nListSeq = C.nListSeq
            INNER JOIN tNormalListCateProduct CP ON (L.nListSeq = CP.nListSeq AND CP.nCategorySeq1 <> 0)
            WHERE L.nBoardSeq IN (60,61,62,63,64,65,66,67,68,130)
            '''
    }
    return sql_dict[key]

def execute_sql(cursor, sql):
    cursor.execute(sql)
    data = cursor.fetchall()
    return data

def make_csv(rows, flag):
    columns = ['L.nListSeq', 'L.nBoardSeq', 'title', 'content', 'category', 'CP.nCategorySeq2', 'CP.nCategorySeq3',
               'CP.nCategorySeq4']
    rows = list(rows)
    for i in range(len(rows)):
        rows[i] = list(rows[i])
    with open('all_data.csv', 'a', encoding='utf-8') as f:
        write_csv = csv.writer(f)
        if flag:
            write_csv.writerow(columns)
        for row in rows:
            write_csv.writerow(row)


if __name__ == '__main__':
    # 1. db 연결
    dbBoard_connect, dbBoard_cursor = db_connect('dbBoard')
    dbBoardPool_connect, dbBoardPool_cursor = db_connect('dbBoardPool')

    # 2. sql문 작성
    sql = return_sql("duplicate_category_news_sql")

    # 3. dbBoard 데이터 프레임화 및 csv 저장
    # 3-1. sql문 적용 : 카테고리 매칭된 뉴스 데이터 샘플 조회
    duplicate_category_news = execute_sql(dbBoard_cursor, sql)
    # 3-2. csv로 저장
    make_csv(duplicate_category_news, True)

    # 4. dbBoardPool 데이터 프레임화 및 csv 저장
    # 4-1. sql문 적용 : 카테고리 매칭된 뉴스 데이터 샘플 조회
    pool_duplicate_category_news = execute_sql(dbBoardPool_cursor, sql)
    # 4-2. csv로 저장
    make_csv(pool_duplicate_category_news, False)

    # 5. 연결 종료
    dbBoard_connect.close()
    dbBoardPool_connect.close()

