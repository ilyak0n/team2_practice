import gradio as gr

from colorflow_core import (
    extract_line_image,
    colorize_image
)

users = {
    "admin": {
        "password": "12345",
        "role": "admin"
    }
}

examples = [
    [
        "./assets/chb03.jpg",
        ["./assets/ref06.jpg", "./assets/ref08.jpg", "./assets/ref09.jpg"]
    ]
]


def func_register(username, password, password2):
    if not username or not password:
        return "!!! Заполните все поля !!!"

    if username in users:
        return "!!! Такой пользователь уже есть !!!"

    if password != password2:
        return "!!! Пароли не совпадают !!!"

    users[username] = {
        "password": password,
        "role": "user"
    }
    return "Успешно! Теперь надо войти"


def func_login(username, password):
    if username not in users:
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "!!! Пользователь не найден !!!",
            "",
            None
        )
    
    user = users[username]

    if user["password"] != password:
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "!!! Вы ввели неверный пароль !!!",
            "",
            None
        )
    
    role = user["role"]
    greeting = (
        f"""
        <br/>
        <div style="text-align: center; font-size: 22px; font-weight: 600;">
            Здравствуйте, <b>{username}</b>
            <br/>
            <span style="font-size: 16px; font-weight: normal;">
                Ваша роль в системе: <i>{role}</i>
            </span>
        </div>
        <br/>
        """
    )
    
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=(role == "admin")),
        "",
        greeting,
        {"username": username, "role": role}
    )


def func_logout():
    return (
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        None
    )


with gr.Blocks() as demo:

    user_state = gr.State(None)

    with gr.Tabs():

        with gr.Tab("Главная страница"):
            gr.HTML(
            """
            <div style="max-width: 900px; margin: auto; text-align: center;">

                <h1>Добро пожаловать!</h1>

                <p style="font-size: 16px;">
                    Данный веб-сервис предназначен для автоматической раскраски черно-белых страниц комиксов.
                </p>

                <hr/>

                <h2>О проекте</h2>
                <p>
                    Проект разработан в рамках учебной проектно-технологической практики Московского Государственного Университета Геодезии и Картографии.
                </p>
                <p>
                    Над проектом работали студенты 3 курса направления прикладной информатики.
                </p>
                <p>
                    Илья - тимлид, frontend-разработчик, ответственный за интерфейс.<br/>
                    Сергей - backend-разработчик, отвественный за работу модели.
                </p>

                <hr/>

                <h2>Краткая инструкция по использованию</h2>
                <ol style="text-align: left; display: inline-block;">
                    <li>Перейдите во вкладку <b>«Зона раскраски»</b>.</li>
                    <li>Загрузите черно-белое изображение.</li>
                    <li>Выберите стиль входного изображения и требуемое разрешение.</li>
                    <li>Нажмите кнопку <b>«Предобработка»</b>.</li>
                    <li>Добавьте референсные изображения.</li>
                    <li>Нажмите кнопку <b>«Раскрасить»</b> и дождитесь результата.</li>
                </ol>

                <hr/>

                <p style="font-style: italic;">
                    Хотим выразить слова благодарности команде разработки нейросети <b>ColorFlow</b>!
                </p>

                <hr/>

                <p style="font-style: italic;">
                    Чтобы сохранять результаты и иметь расширенный доступ, рекомендуем зарегистрироваться.<br/>
                    Приятной работы!
                </p>

            </div>
            """
            )

        with gr.Tab("Зона раскраски"):
            VAE_input = gr.State()
            input_context = gr.State()

            with gr.Row():
                input_style = gr.Radio(
                    ["GrayImage(ScreenStyle)", "Sketch_Shading", "Sketch"],
                    label="Стиль входного изображения",
                    value="GrayImage(ScreenStyle)"
                )

            with gr.Row():
                inp_img = gr.Image(
                    type="pil",
                    label="Загрузить изображение (ч/б / скетч)"
                )
                extracted_img = gr.Image(
                    type="pil",
                    label="Предобработанное изображение"
                )

            resolution = gr.Radio(
                ["640x640", "512x800", "800x512"],
                label="Разрешение",
                value="640x640"
            )

            preprocess_btn = gr.Button("Предобработка")

            preprocess_btn.click(
                extract_line_image,
                inputs=[inp_img, input_style, resolution],
                outputs=[extracted_img, VAE_input, input_context]
            )

            ref_imgs = gr.Files(
                label="Референсы (несколько)",
                file_count="multiple"
            )

            seed = gr.Slider(0, 100000, value=0, step=1, label="Seed")
            steps = gr.Slider(4, 50, value=10, step=1, label="Шаги генерации")

            color_btn = gr.Button("🎨 Раскрасить")

            outp_img = gr.Gallery(
                type="pil",
                label="Результат"
            )

            color_btn.click(
                colorize_image,
                inputs=[
                    VAE_input,
                    input_context,
                    ref_imgs,
                    resolution,
                    seed,
                    input_style,
                    steps
                ],
                outputs=outp_img
            )

        with gr.Tab("Вход", visible=True) as tab_login:
            login_username = gr.Textbox(label="Введите логин")
            login_password = gr.Textbox(label="Введите пароль", type="password")
            login_btn = gr.Button("Войти")
            login_msg = gr.Markdown()

        
        with gr.Tab("Регистрация", visible=True) as tab_registration:
            reg_username = gr.Textbox(label="Придумайте логин")
            reg_password = gr.Textbox(label="Придумайте пароль", type="password")
            reg_password2 = gr.Textbox(
                label="Повторите пароль", type="password"
            )
            reg_btn = gr.Button("Зарегистрироваться")
            reg_msg = gr.Markdown()

        
        with gr.Tab("Личный кабинет", visible=False) as tab_account:
            welcome_msg = gr.HTML()

            logout_btn = gr.Button("Выйти")
        
        with gr.Tab("Панель управления", visible=False) as tab_admin:
            gr.HTML('')
        
        reg_btn.click(
            func_register,
            inputs=[reg_username, reg_password, reg_password2],
            outputs=reg_msg
        )

        login_btn.click(
            func_login,
            inputs=[login_username, login_password],
            outputs=[
                tab_login,
                tab_registration,
                tab_account,
                tab_admin,
                login_msg,
                welcome_msg,
                user_state
            ]
        )

        logout_btn.click(
            func_logout,
            outputs=[
                tab_login,
                tab_registration,
                tab_account,
                tab_admin,
                user_state
            ]
        )


demo.launch(inbrowser=True, share=True)
