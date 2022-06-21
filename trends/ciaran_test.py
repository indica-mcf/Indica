# import mysql.connector
# import numpy as np
# #import pymysql
# #from trends.trends_database import Database
#
# mysql_conn = mysql.connector.connect(
#     user="marco.sertoli",
#     password='Marco3142!',
#     host='192.168.1.9',
#     port=3306
# )
#
# mycursor = mysql_conn.cursor()
#
# # Creating a database
# mycursor.execute("CREATE DATABASE testdatabase")
#
# #Creating a table
# mycursor.execute("CREATE TABLE Person (name VARCHAR(50), age smallint UNSIGNED, personID int PRIMARY KEY AUTO_INCREMENT)")
#
# """
# VARHCAR - variable character = string up to 50 (in this case)
# PRIMARY KEY - unique values associated with each row in the table (for St40, this would be the pulse number since no two are identical)
# AUTO_INCREMENT - automatically creates a primary key different from previous primary key in the table
# """
#
# mycursor.execute("DESCRIBE Person")
#
# """
# DESCRIBE person will give you the information from the Person table in the format:
#
# """
#
# for x in mycursor:
#     print(x)
#
# # Adding elements into table and retrieving them
#
# mycursor.execute("INSERT INTO Person (name,age) VALUES ('me', 21)")
#
# """
# This is not a good way of doing it, to do it safely, string formatting is a safer way, allows us to pass in variables
# """
#
# mycursor.execute("INSERT INTO Person (name, age) VALUES (%,%)", ("me", "21"))
#
# # Committing changes - saved permenantly
# #mysql_conn.commit()
#
# # Looking at values inside the table
#
# mycursor.execute("SELECT * FROM Person")
# """
# * Gives us all the information inside the table 'Person', to access this information we need to loop through the table
# Initially the loop will only return ('me', '21') but if we write a statement to insert another entry, we will have 2 as they're
# saved.
# """
#
# for x in mycursor:
#     print(x)
#
# # Starting from a clean table
#
# mycursor.execute("CREATE TABLE test (name varchar(50) NOT NULL, created datetime NOTNULL, gender ENUM('M', 'F', 'O') NOT NUL, ID int PRIMARY KEY NOT NULL AUTO_INCREMENT")
#
# #mycursor.execute("INSERT INTO Test (name, created, gender) VALUES (%s,%s,%s)", ("me", datetime.now(), "M"))
# #mysql_conn.commit()
#
#
# # More on select command
#
# mycursor.execute("SELECT * FROM Test WHERE gender = 'M'")
# for x in mycursor:
#     print(x)
#
#
# # Ordering entries
# mycursor.execute("SELECT * FROM Test WHERE gender = 'M' ORDER BY id ASC")
#
# """
# We can order the selected column in a number of ways, e.g., ASC (ascending) and DESC (descending)
# If you want to select from a specific column * -> id, name
# """
# mycursor.execute("SELECT id, name FROM Test WHERE gender = 'M' ORDER BY id DESC")
#
# # Modifying tables
#
# mycursor.execute("ALTER TABLE Test ADD COLUMN food VARCHAR(50) NOT NULL")
#
# mycursor.execute("DESCRIBE Test")
# print(mycursor.fetchone())
# for x in mycursor:
#     print(x)
#
# # Remove column
# mycursor.execute("ALTER TABLE Test DROP food")
#
# # Changing column name
# mycursor.execute("ALTER TABLE Test CHANGE name first_name VARCHAR(50)")
#
# ###
# # Foreign Keys
# """
# Allows you to reference a table from another table
# """
#
#
# #mysql_cursor = mysql_conn.cursor()
# #sql = "SELECT * FROM regression_database"
# #mysql_cursor.execute(sql)
#
#
#
# """
# Examples of reading data from mySQL
# """
# # Reading in all data in ascending pulse number
# Q1 = "SELECT pulseNo FROM regression_database ORDER BY pulseNo ASC"
#
# # Finding data for a particular pulse
# Q2 = "SELECT * from regression_database WHERE pulseNo = 10014"
#
# # Returning plasma current for a particular pulse
# Q3 = "SELECT ipla_efit from regression_database WHERE pulseNo = 10014"
#
# # Retrieving data where max plasma current is achieved
# Q4 = "SELECT pulseNos, ipla_efit, wp_efit FROM regression_database WHERE " \
#      "ipla_efit=(SELECT MAX(ipla_efit)) FROM regression_database"
#
# mycursor.execute(Q1)

variable = 'ipla_efit'
json_var = variable.replace('_', '#')

print(json_var)

def read_from_mysql(querie, key: str = None, variable: str = None, parameter: str = None):
    """
    Reads data from regression database

    Parameters
    ----------
    parameter
    variable
    querie
    key

    Base structure:
    Database  = {'binned': {            (key)
                            'time': []  (variable)
                            'ip#efit': {
                                        'data': Ip    (parameter)
                                        'gradient': Ip_gradient}},

                'static': {
                            'max_val': {
                                        'ip#efit': {
                                                    data': Ip,
                                                    'error_lower': Ip_error}}}
                            'pulseNo': number



    """

    pymysql_connector = pymysql.connect(
        user='marco.sertoli',
        password='Marco3142!',
        host='192.168.1.9',
        database='st40_test',
        port=3306,
        cursorclass=pymysql.cursors.DictCursor
    )
    with pymysql_connector:
        with pymysql_connector.cursor() as cursor:
            sql = querie
            cursor.execute(sql)
            result = cursor.fetchone()
            data = json.loads(result['data'])

            # print(type(data))
            # print(data['static']['max_val'])

            if key == 'static':
                if parameter is not None:
                    return data[key][variable][parameter], 'yes'
                else:
                    return data[key][variable]

            if key == 'binned':
                if parameter is not None:
                    return data[key][variable][parameter], 'yes'
                else:
                    return data[key][variable]

def test_read_from_mysql(query, key: str = None, variable: str = None, parameter: str = None):
    """
    Reads data from regression database

    Parameters
    ----------
    parameter
    variable
    query
    key

    Base structure:
    Database  = {'binned': {            (key)
                            'time': []  (variable)
                            'ip#efit': {
                                        'data': Ip    (parameter)
                                        'gradient': Ip_gradient}},

                'static': {
                            'max_val': {
                                        'ip#efit': {
                                                    data': Ip,
                                                    'error_lower': Ip_error}}}
                            'pulseNo': number}}
    """

    pymysql_connector = pymysql.connect(
        user='marco.sertoli',
        password='Marco3142!',
        host='192.168.1.9',
        database='st40_test',
        port=3306,
        cursorclass=pymysql.cursors.DictCursor
    )

    with pymysql_connector:
        with pymysql_connector.cursor() as cursor:
            sql = query
            cursor.execute(sql)
            result = cursor.fetchone()
            return result
