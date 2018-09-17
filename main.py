# https://stackoverflow.com/a/27827341
# https://vinta.ws/code/converting-doc-to-pdf-using-unoconv-and-python.html
# dipgen/bg.png
# libreoffice --headless --convert-to png --outdir ./pdfs sample1.pdf
# libreoffice --headless --convert-to png --outdir ./pdfs ./pdfs/*.docx
# PyPDF2 вместо текста читает какой-то мусор
# docxtpl крутая штука. jinja-шаблоны прям в docx
# pymorphy2 - умеет угадывать падеж слова, и ставить его в другой
# https://stackoverflow.com/questions/30349542/command-libreoffice-headless-convert-to-pdf-test-docx-outdir-pdf-is-not
# https://stackoverflow.com/questions/34617422/how-to-optimize-image-size-using-wand-in-python
# конвертнуть между форматами:
#   sudo apt install imagemagick && convert image.jpg image.png
# уменьшить png:
#   pngquant -o dip_min.png --force --quality=10-20 dip.png

import os
import csv
import re
import pathlib
import shutil
import subprocess
from contextlib import contextmanager
from pprint import pprint

import funcy
from funcy import suppress
import pymorphy2
from tqdm import tqdm
from docxtpl import DocxTemplate

from natasha import NamesExtractor
from collections import OrderedDict
from slugify import slugify
from transliterate import translit
from collections import defaultdict
import itertools

morph = pymorphy2.MorphAnalyzer()


@contextmanager
def cd(newdir):
    # with cd(<folder>):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def text_to_genitive(text):
    def words_to_genitive(words):
        do_transform = True
        for i, word in enumerate(words):
            parse = morph.parse(word)[0]
            if do_transform and parse.tag.case == 'nomn' and \
               not word.isupper() and word[0] != '"' and \
               not (i != 0 and word[0].isupper()):
                yield parse.inflect({'gent'}).word
            else:
                do_transform = False
                yield word
    # words = [parse_to_genitive(word, morph.parse(word)[0]) for word in text.split()]
    return ' '.join(words_to_genitive(text.split()))


def name_to_gender(first_name, family_name=None):

    # пытаемся определить пол по фамилии
    if family_name is not None:
        # если определит фамилию как родительный падеж - значит баба xD
        # upd.: Латанов Владислав стал первым контрпримером
        # if morph.parse(family_name)[0].tag.case == 'gent':
        #     return 0
        # пытаемся детектировать мужские фамилии по окончанию
        suffixes = ['ов', 'ев', 'ёв', 'ин', 'ын', 'ский', 'цкий']
        if funcy.any(lambda suff: family_name.endswith(suff), suffixes):
            return 1
        # пытаемся детектировать женские фамилии по окончанию
        suffixes = ['ова', 'ева', 'ёва', 'ина', 'ына', 'ская', 'цкая']
        if funcy.any(lambda suff: family_name.endswith(suff), suffixes):
            return 0

    # пытаемся определить пол по имени автоматически
    parse = morph.parse(first_name)[0]
    score = parse.score
    gender = parse.tag.gender == 'masc'
    if score > 0.9:
        return gender

    # если не получилось - выводим имя и детектированный пол (для проверки глазами),
    # и пытаемся определить вручную (как с фамилиями)
    print(f'{first_name} {family_name} - ', end='')
    # мужские имена
    suffixes = ['он', 'Максим', 'Евгений', 'Владислав']
    if funcy.any(lambda suff: first_name.endswith(suff), suffixes):
        print('кунец?')
        return 1
    # женские имена
    suffixes = ['на', 'Варвара']
    if funcy.any(lambda suff: first_name.endswith(suff), suffixes):
        print('тянка?')
        return 0

    print('ЧТО ЗА ПОКЕМОН?')
    raise NameError


def school_to_genitive(school):
    if school.isdigit():
        return 'школы № %s' % school
    return text_to_genitive(school)


def generate_from_template(docx_template, out_dir, namelines_list, contexts_list):
    for nameline, context in tqdm(list(zip(namelines_list, contexts_list)), disable=False):

        nameline = slugify(nameline)
        doc = DocxTemplate(docx_template)

        with cd(out_dir):

            # подставляем свои данные в шаблон формата docx
            doc.render(context)
            doc.save('docx/%s.docx' % nameline)

            # перегоняем docx в pdf
            # чтобы команда работала, libreoffice должен быть закрыт(!)
            # https://stackoverflow.com/a/43907693
            subprocess.call(['libreoffice', '--headless',
                                            '--convert-to', 'pdf',
                                            '--outdir', './pdf',
                                            './docx/%s.docx' % nameline],
                            stdout=subprocess.DEVNULL)

            # создаем превьюшки в png
            # если установлен wand и он не падает
            # (из-за отсутствия libmagickwand-dev ghostscript)
            with funcy.suppress(ImportError):
                from wand.image import Image
                from wand.color import Color
                from wand.exceptions import WandError
                with funcy.suppress(WandError):
                    with Image(filename='pdf/%s.pdf' % nameline, resolution=300, background=Color('white')) as img:
                        img.save(filename='png/%s.png' % nameline)


def recreate_output_directory(out_dir):
    # все складывается в папку generated в текущей директории
    # перед запуском программы очищаем ее, и пересоздаем директории
    shutil.rmtree(out_dir, ignore_errors=True)
    for d in 'docx', 'pdf', 'png':
        pathlib.Path('%s/%s' % (out_dir, d)).mkdir(parents=True, exist_ok=True)


def cleanup_output_directory(out_dir):
    # после всей работы можно удалить временные файлы
    for d in 'docx', :
        shutil.rmtree('%s/%s' % (out_dir, d))


def generate_with_cleanup(docx_template, out_dir, namelines_list, contexts_list):
    recreate_output_directory(out_dir)
    generate_from_template(docx_template, out_dir, namelines_list, contexts_list)
    cleanup_output_directory(out_dir)


def keys_to_lower(dic):
    return funcy.walk_keys(lambda k: k.lower(), dic)

def trim_values(dic):
    return funcy.walk_values(lambda v: v.strip(), dic)

def get_from_csv(path):
    with open(path, 'r') as f:
        return [trim_values(keys_to_lower(row)) for row in csv.DictReader(f)]


def str_to_int(s, default=0):
    with funcy.supress(ValueError):
        return int(s)
    return default


def merge_rows_on_key(l1, l2, key):
    """
    https://stackoverflow.com/a/5501893
    """
    d = defaultdict(dict)
    for elem in itertools.chain(l1, l2):
        d[elem[key]].update(elem)
    return d.values()

def shorten_school(school):
    # пытаемся извлечь номер: если он больше 5, то возвращаем его
    # (чтобы не вернуть "1" на какой-нибудь "1-й МОК")
    nums = re.findall(r"\d+", school)
    if nums and int(nums[0]) > 5:
        return nums[0]
    # удаляем слова ГБОУ, ГАОУ и прочие Образовательные Учереждения
    school = ' '.join([word for word in school.split() if not (word.endswith('ОУ') and word == word.upper())])
    return school


#########################################################
# ДИПЛОМЫ ДЛЯ КОНСТРУИРОВАНИЯ

rows = get_from_csv('input/2018.07.21_konstruirovanie/SCH_spisok_21apr(1).csv')
teams = set((row['команда'] for row in rows))

def get_team_rows(team):
    return [row for row in rows if row['команда'] == team]

def get_team_field(team, field):
    rows = get_team_rows(team)
    return [row[field] for row in rows if row[field]]

def get_team_names(team):
    return ', '.join(f'{fam} {name}' for fam, name in zip(get_team_field(team, 'фамилия'), get_team_field(team, 'имя')))

contexts_list = [{**row,
                  'степень': 'I' * int(get_team_field(row['команда'], 'место')[0]),
                  'класс': re.findall(r"\d+", row['класс'])[0],
                  'учащийся': 'учащийся' if name_to_gender(row['имя'], row['фамилия']) else 'учащаяся',
                  'школа_р': school_to_genitive(shorten_school(row['школа'])),
                  'name_line': '%s__%s_%s' % (shorten_school(row['школа']), row['фамилия'], row['имя']),
                  } for row in rows if get_team_field(row['команда'], 'место')]

generate_with_cleanup(docx_template='input/2018.07.21_konstruirovanie/winner.docx', out_dir='generated/winners',
                      namelines_list=[translit(d['name_line'], 'ru', reversed=True) for d in contexts_list],
                      contexts_list=contexts_list)

generate_with_cleanup(docx_template='input/2018.07.21_konstruirovanie/winner_blank.docx', out_dir='generated/winners_blank',
                      namelines_list=[translit(d['name_line'], 'ru', reversed=True) for d in contexts_list],
                      contexts_list=contexts_list)

contexts_list = [{**row,
                  'класс': re.findall(r"\d+", row['класс'])[0],
                  'номинация': get_team_field(row['команда'], 'номинация')[0],
                  'учащийся': 'учащийся' if name_to_gender(row['имя'], row['фамилия']) else 'учащаяся',
                  'школа_р': school_to_genitive(shorten_school(row['школа'])),
                  'name_line': '%s__%s_%s' % (shorten_school(row['школа']), row['фамилия'], row['имя']),
                  } for row in rows if get_team_field(row['команда'], 'номинация')]

generate_with_cleanup(docx_template='input/2018.07.21_konstruirovanie/nominee.docx', out_dir='generated/nominees',
                      namelines_list=[translit(d['name_line'], 'ru', reversed=True) for d in contexts_list],
                      contexts_list=contexts_list)

generate_with_cleanup(docx_template='input/2018.07.21_konstruirovanie/nominee_blank.docx', out_dir='generated/nominees_blank',
                      namelines_list=[translit(d['name_line'], 'ru', reversed=True) for d in contexts_list],
                      contexts_list=contexts_list)


#########################################################


#########################################################
# ДИПЛОМЫ ДЛЯ ЗИМНЕГО РОБОФУТБОЛА
"""
rows = get_from_csv('input/2018.07.21_robofootball_zima/robofootball_dec2017.csv')

contexts_list = [{**row,
                  'фамилии_имена': ', '.join(set(
                    funcy.keep(('%s %s' %(row[f'фамилия {i}'], row[f'имя {i}'])).strip() for i in range(1, 4))
                  )),
                  'степень': 'I' * int(row['место']),
                  'школа_р': school_to_genitive(shorten_school(row['школа'])),
                  'name_line': '%s__%s' % (shorten_school(row['школа']), row['команда']),
                  } for row in rows if row['место']]

generate_with_cleanup(docx_template='input/2018.07.21_robofootball_zima/Робофутбол-Зима_Диплом.docx', out_dir='generated',
                      namelines_list=[translit(d['name_line'], 'ru', reversed=True) for d in contexts_list],
                      contexts_list=contexts_list)
"""
#########################################################


#########################################################
# ДИПЛОМЫ ДЛЯ ШАГА В БУДУЩЕЕ
"""
rows = get_from_csv('input/2018.07.21_moy_shag/data.csv')

contexts_list = [{**row,
                  'фамилии_имена': ', '.join(set(
                    funcy.keep(('%s %s' %(row[f'фамилия {i}'], row[f'имя {i}'])).strip() for i in range(1, 3))
                  )),
                  'степень': 'I' * int(row['место']),
                  'школа_р': school_to_genitive(shorten_school(row['школа'])),
                  'name_line': '%s__%s' % (shorten_school(row['школа']), row['команда']),
                  } for row in rows if row['место']]

generate_with_cleanup(docx_template='input/2018.07.21_moy_shag/template.docx', out_dir='generated',
                      namelines_list=[translit(d['name_line'], 'ru', reversed=True) for d in contexts_list],
                      contexts_list=contexts_list)
"""
#########################################################


#########################################################
# ДИПЛОМЫ ДЛЯ ВЕСЕННЕГО РОБОФУТБОЛА (24 АПРЕЛЯ 2018)
"""
rows = get_from_csv('input/2018.07.21_robofootball_vesna/robofootball_apr2018.csv')

contexts_list = [{**row,
                  'фамилии_имена': ', '.join(set(
                    [' '.join(s.split()[:2]) for s in funcy.keep(row['участник %s' % i] for i in range(1, 4))]
                  )),
                  'степень': 'I' * int(row['место']),
                  'школа_р': school_to_genitive(row['школа']),
                  'строка_с_категорией': 'в зрительской категории' if row['категория'] == 'Зрит' else '',
                  'name_line': '%s__%s' % (row['школа'], row['команда']),
                  } for row in rows if row['место']]

generate_with_cleanup(docx_template='input/2018.07.21_robofootball_vesna/Робофутбол-Весна_Диплом.docx', out_dir='generated',
                      namelines_list=[translit(d['name_line'], 'ru', reversed=True) for d in contexts_list],
                      contexts_list=contexts_list)
"""
#########################################################


#########################################################
# ДИПЛОМЫ ДЛЯ РОБОТСАМ 2.0 (ПОБЕДИТЕЛИ И ПО НОМИНАЦИЯМ)
"""
rows = get_from_csv('input/2018.06.05_rsam2_dip/data.csv')
totals = get_from_csv('input/2018.06.05_rsam2_dip/winners.csv')

rows = merge_rows_on_key(totals, rows, 'команда')

contexts_list = [{**row,
                  'фамилии_имена': ', '.join(sorted(set(funcy.keep(row['участник %s' % i] for i in range(1, 6))))),
                  'степень': row['место'],
                  'школа_р': school_to_genitive(row['школа']),
                  'номинация': row['возможная номинация'],
                  'name_line': '%s__%s' % (row['школа'], row['команда']),
                  } for row in rows if row['место'] or row['возможная номинация']]

contexts_winners = [row for row in contexts_list if row['место']]
contexts_nominees = [row for row in contexts_list if row['возможная номинация']]

generate_with_cleanup(docx_template='input/2018.06.05_rsam2_dip/template.docx', out_dir='generated/winners',
                      namelines_list=[translit(d['name_line'], 'ru', reversed=True) for d in contexts_winners],
                      contexts_list=contexts_winners)


generate_with_cleanup(docx_template='input/2018.06.05_rsam2_dip/template_enc.docx', out_dir='generated/nominees',
                      namelines_list=[translit(d['name_line'], 'ru', reversed=True) for d in contexts_nominees],
                      contexts_list=contexts_nominees)

generate_with_cleanup(docx_template='input/2018.06.05_rsam2_dip/template_blank.docx', out_dir='generated/winners_blank',
                      namelines_list=[translit(d['name_line'], 'ru', reversed=True) for d in contexts_winners],
                      contexts_list=contexts_winners)

generate_with_cleanup(docx_template='input/2018.06.05_rsam2_dip/template_enc_blank.docx', out_dir='generated/nominees_blank',
                      namelines_list=[translit(d['name_line'], 'ru', reversed=True) for d in contexts_nominees],
                      contexts_list=contexts_nominees)
"""
#########################################################


#########################################################
# ДИПЛОМЫ ДЛЯ РОБОТСАМ_1
"""
rows = get_from_csv('input/2018.06.05_rsam1_dip/data.csv')

contexts_list = [{**row,
                  'фамилии_имена': ', '.join(set(funcy.keep(row['участник %s' % i] for i in range(1, 6)))),
                  'степень': 'I' * int(row['место']),
                  'группа': ('%s класс' % row['группа']) if len(row['группа']) == 1 else ('%s классы' % row['группа']),
                  'школа_р': school_to_genitive(row['школа']),
                  'name_line': '%s__%s' % (row['школа'], row['команда']),
                  } for row in rows if row['место']]

generate_with_cleanup(docx_template='input/2018.06.05_rsam1_dip/template.docx', out_dir='generated/bg',
                      namelines_list=[translit(d['name_line'], 'ru', reversed=True) for d in contexts_list],
                      contexts_list=contexts_list)

generate_with_cleanup(docx_template='input/2018.06.05_rsam1_dip/template_blank.docx', out_dir='generated/blank',
                      namelines_list=[translit(d['name_line'], 'ru', reversed=True) for d in contexts_list],
                      contexts_list=contexts_list)
"""
#########################################################
