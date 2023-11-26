from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import data_loading
import data_preprocessing
import model_creation
import model_testing


def StepOne(ti):
    df_x_train, df_y_train, df_x_test, df_y_test = data_loading.load()

    ti.xcom_push("df_x_train", df_x_train)
    ti.xcom_push("df_y_train", df_y_train)
    ti.xcom_push("df_x_test", df_x_test)
    ti.xcom_push("df_y_test", df_y_test)


def StepTwo(ti):
    df_x_train = ti.xcom_pull(task_ids="task_1", key="df_x_train")
    df_y_train = ti.xcom_pull(task_ids="task_1", key="df_y_train")
    df_x_test = ti.xcom_pull(task_ids="task_1", key="df_x_test")
    df_y_test = ti.xcom_pull(task_ids="task_1", key="df_y_test")

    df_x_train, df_y_train, df_x_test, df_y_test = \
        data_preprocessing.preprocess(df_x_train, df_y_train, df_x_test, df_y_test)

    ti.xcom_push("df_x_train_preprocessing", df_x_train)
    ti.xcom_push("df_y_train_preprocessing", df_x_train)
    ti.xcom_push("df_x_test_preprocessing", df_x_train)
    ti.xcom_push("df_y_test_preprocessing", df_x_train)


def StepThree(ti):
    df_x_train = ti.xcom_pull(task_ids="task_2", key="df_x_train_preprocessing")
    df_y_train = ti.xcom_pull(task_ids="task_2", key="df_y_train_preprocessing")

    model = model_creation.create_model(df_x_train, df_y_train)
    ti.xcom_push("model", model)


def StepFour(ti):
    df_x_test = ti.xcom_pull(task_ids="task_2", key="df_x_test_preprocessing")
    df_y_test = ti.xcom_pull(task_ids="task_2", key="df_y_test_preprocessing")
    model = ti.xcom_pull(task_ids="task_3", key="model")

    score = model_testing.test_model(model, df_x_test, df_y_test)

    ti.xcom_push("score", score)


args = {
    'owner': 'admin',
    'start_date': datetime(year=2023, day=26, month=11)
}

with DAG('My script', description='My mlops pipeline', schedule_interval='*/1 * * * *', catchup=False,
         default_args=args) as dag:  # 0 * * * *   */1 * * * *

    task_1 = PythonOperator(task_id="task_1", python_callable=StepOne)
    task_2 = PythonOperator(task_id="task_2", python_callable=StepTwo)
    task_3 = PythonOperator(task_id="task_3", python_callable=StepThree)
    task_4 = PythonOperator(task_id="task_4", python_callable=StepFour)
