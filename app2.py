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
        f"Привет, {username}" if role == "user" else f"Здравствуйте, {username}"
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
                <div style="text-align: center;">
                    <h1>В будущем это будет главная страница с большим описанием</h1>
                    <i>Все в будущем...</i>
                    </br>
                    <h2>Также здесь планируется добавить небольшую памятку-инструкцию о том, как пользоваться нашим приложением</h2>
                    <i>Или лучше это сделать на следующей вкладке...</i>
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
            welcome_msg = gr.Markdown()

            gr.HTML(
                """
                <div style="text-align: center;">
                    <h1>Здесь будет страница с личным кабинетом</h1>
                    <i>Совсем скоро...</i>
                </div>
                """
            )

            logout_btn = gr.Button("Выйти")
        
        with gr.Tab("Панель управления", visible=False) as tab_admin:
            gr.HTML(
                """
                <div style="text-align: center;">
                    <h1>Тут админ сможеть смотреть за другими пользователями</h1>
                    <i>До реализации еще чуть-чуть надо потерпеть...</i>
                </div>
                """
            )

        
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
