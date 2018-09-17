# Генератор дипломов, сертификатов и грамот.

Требует наличия docker.
Если запускать вне докера - требует libreoffice (полный список зависимостей - в Dockerfile). При работе скрипта вне контейнера libreoffice не должен использоваться (иначе вылезет ошибка).

## Принцип работы

Скрипт main.py:
* читает csv-файл
* берет оттуда имена участников конкурса, их классы и прочее
* и заполняет ими размеченный в формате jinja2 docx-шаблон
* из получившихся docx-файлов с заполненными полями генерируются pdf-грамоты
* для удобства просмотра также генерируются png-версии грамот

Если необходимо сгенерировать грамоту "с подложкой":
* вставить картинку в docx-шаблон (Insert > Image)
* растянуть на полную (LKM > Properties > Crop > сделать width и height 100%)
* поставить в фон (LKM > Wrap > In Background)

Перед повторныйм запуском скрипта очищать папку generated не обязательно - скрипт это делает автоматически.

## Запуск в контейнере

Для удобства есть скриптик start.sh, запускающий все это дело в контейнере. Некоторая его (и Dockerfile'а) сложность связана с тем, что при монтировании папок файлы, созданные рутом внутри контейнера, остаются принадлежать руту же и снаружи контейнера. Чтобы не юзать chown, а уже в процессе генерации иметь нормальные данные, в контейнере создается и используется пользователь с такимя user_id и group_id, как и на хостовой машине.
Также: скрипт вначале запуска удаляет dangling образы. Это связано с тем, что образ с либреоффисом тяжелый, и при разработке если каждый раз не удалять - они быстро накапливаются.