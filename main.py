import os
from typing import Dict, List, Tuple, Literal, TypedDict
from unicodedata import normalize, east_asian_width

import deepl
import googletrans
from langdetect import detect
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from termcolor import COLORS, colored


Tag = str
Token = str


class TaggedTokens(TypedDict):
    text: str
    token_tag_pairs: List[Tuple[Token, Tag]]


class MorphologicalAnalyzer:
    tag_table = {
        "CC": ["Coordinating conjunction", "調整接続詞"],
        "CD": ["Cardinal number", "基数"],
        "DT": ["Determiner", "限定詞"],
        "EX": ["Existential there", "存在を表す there"],
        "FW": ["Foreign word", "外国語"],
        "IN": ["Preposition or subordinating conjunction", "前置詞|従属接続詞"],
        "JJ": ["Adjective", "形容詞"],
        "JJR": ["Adjective, comparative", "形容詞(比較級)"],
        "JJS": ["Adjective, superlative", "形容詞(最上級)"],
        "LS": ["List item marker", "-"],
        "MD": ["Modal", "法"],
        "NN": ["Noun, singular or mass", "名詞"],
        "NNS": ["Noun, plural", "名詞(複数形)"],
        "NNP": ["Proper noun, singular", "固有名詞"],
        "NNPS": ["Proper noun, plural", "固有名詞(複数形)"],
        "PDT": ["Predeterminer", "前限定辞"],
        "POS": ["Possessive ending", "所有格の終わり"],
        "PRP": ["Personal pronoun", "人称代名詞"],
        "PRP$": ["Possessive pronoun", "所有代名詞"],
        "RB": ["Adverb", "副詞"],
        "RBR": ["Adverb, comparative", "副詞(比較級)"],
        "RBS": ["Adverb, superlative", "副詞(最上級)"],
        "RP": ["Particle", "不変化詞"],
        "SYM": ["Symbol", "記号"],
        "TO": ["to", "前置詞 to"],
        "UH": ["Interjection", "感嘆詞"],
        "VB": ["Verb, base form", "動詞(原形)"],
        "VBD": ["Verb, past tense", "動詞(過去形)"],
        "VBG": ["Verb, gerund or present participle", "動詞(動名詞|現在分詞)"],
        "VBN": ["Verb, past participle", "動詞(過去分詞)"],
        "VBP": ["Verb, non-3rd person singular present", "動詞(三人称単数以外の現在形)"],
        "VBZ": ["Verb, 3rd person singular present", "動詞(三人称単数の現在形)"],
        "WDT": ["Wh-determiner", "Wh 限定詞"],
        "WP": ["Wh-pronoun", "Wh 代名詞"],
        "WP$": ["Possessive wh-pronoun", "所有 Wh 代名詞"],
        "WRB": ["Wh-adverb", "Wh 副詞"],
        ",": [",", ","],
        ".": [".", "."],
    }

    def __init__(
        self, tag_type: Literal["abbreviation", "expansion", "japanese"] = None
    ) -> None:
        self.tag_type = "abbreviation" if tag_type is None else tag_type

    def analyze(self, texts: str) -> List[TaggedTokens]:
        texts: List[str] = sent_tokenize(texts)

        result = []
        for text in texts:
            tokens = self.tokenize(text)
            tagged_tokens = nltk.pos_tag(tokens)

            if self.tag_type == "abbreviation":
                result.append(TaggedTokens(text=text, token_tag_pairs=tagged_tokens))
                continue

            index = 0 if self.tag_type == "expansion" else 1
            expression = lambda item: (item[0], self.tag_table[item[1]][index])
            tagged_tokens = list(map(expression, tagged_tokens))
            result.append(TaggedTokens(text=text, token_tag_pairs=tagged_tokens))

        return result

    @staticmethod
    def setup() -> None:
        nltk.download("punkt")
        nltk.download("averaged_perceptron_tagger")

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return word_tokenize(text)


class TextsFormatter:
    def __init__(
        self,
        is_list: bool = False,
        is_translate: bool = False,
        language_code: str = None,
        color_enabled: bool = False,
        colors_to_remove: List[str] = [],
    ) -> None:
        self.is_list = is_list
        self.is_translate = is_translate
        self.color_enabled = color_enabled
        self.language_code = language_code
        self.colors_to_remove = colors_to_remove

    def format_token_tag_pairs(
        self, token_tag_pairs: List[Tuple[Token, Tag]]
    ) -> List[str]:
        # Create an object where tag is key and color is value
        tags = list(set([tag for _, tag in token_tag_pairs]))
        color_tag = self.generate_color_tag(tags, self.colors_to_remove)

        tag_list = set()
        tokens, under_line, part_of_speech_line = "", "", ""

        color = ""
        for token, tag in token_tag_pairs:
            tag_length = self.adjust_length_ja_en(tag)
            if self.color_enabled:
                color = color_tag[tag]
                tag = colored(tag, color)

            # Tag and part of speech
            adjusted_length = len(token)
            if not self.is_list:
                adjusted_length = max(adjusted_length, tag_length)
                if tag_length < adjusted_length:
                    tag += abs(tag_length - adjusted_length) * " "
            else:
                tag_list.add(tag)
            tokens += f" {token.ljust(adjusted_length, ' ')}"
            part_of_speech_line += f" {tag}"

            # Token under line
            token_under_line = adjusted_length * "▔"
            if self.color_enabled:
                token_under_line = colored(token_under_line, color)
            under_line += f" {token_under_line}"

        if self.is_list:
            tag_list = [f"- {tag}" for tag in sorted(tag_list, key=len)]
            part_of_speech_line = "\n".join(tag_list)

        return [tokens, under_line, part_of_speech_line]

    def format(self, texts_data: List[TaggedTokens]) -> str:
        formatted_texts = []
        for text_data in texts_data:
            token_tag_pairs = text_data.get("token_tag_pairs")
            text = text_data.get("text")

            formatted_text = self.format_token_tag_pairs(token_tag_pairs)
            if self.is_translate:
                translated_text = self.translate(text)
                formatted_text.append(f"\n{translated_text}")

            terminal_width = os.get_terminal_size().columns
            formatted_text.append(f"\n{terminal_width * '─'}\n")
            formatted_texts.append("\n".join(formatted_text))

        return "\n".join(formatted_texts)

    def translate(self, text: str) -> str:
        translator = googletrans.Translator()
        if self.language_code is None:
            return text
        translated_text = translator.translate(text=text, dest=self.language_code).text
        return translated_text

    @staticmethod
    def adjust_length_ja_en(ja_string: str) -> int:
        ja_characters = list(normalize("NFKC", ja_string))

        adjusted_length = 0
        for character in ja_characters:
            if east_asian_width(character) == "W":
                adjusted_length += 2
                continue
            adjusted_length += 1

        return int(adjusted_length)

    @staticmethod
    def generate_color_tag(
        tags: List[str], colors_to_remove: List[str] = []
    ) -> Dict[str, str]:
        colors = [color for color in COLORS.keys() if color not in colors_to_remove]
        result = {tag: color for tag, color in zip(tags, colors)}
        return result


class TextsFormatterDeepL(TextsFormatter):
    def __init__(
        self,
        api_key: str,
        is_list: bool = False,
        is_translate: bool = False,
        language_code: str = None,
        color_enabled: bool = False,
        colors_to_remove: List[str] = [],
    ) -> None:
        super().__init__(
            is_list, is_translate, language_code, color_enabled, colors_to_remove
        )
        self.api_key = api_key

    def translate(self, text: str) -> str:
        translator = deepl.Translator(self.api_key).translate_text
        if self.language_code is None:
            return text

        source_lang = detect(text)
        translated_text = translator(
            text=text, source_lang=source_lang, target_lang=self.language_code
        )
        return str(translated_text)


class InteractiveMode:
    def __init__(
        self,
        texts_formatter: TextsFormatter,
        morphological_analyzer: MorphologicalAnalyzer,
    ) -> None:
        self.running = False
        self.texts_formatter = texts_formatter
        self.morphological_analyzer = morphological_analyzer

    def start(self) -> None:
        self.running = True
        print("Interactive mode has been initiated.")

    def stop(self) -> None:
        self.running = False
        print("Interactive mode has been terminated.")

    def help(self) -> None:
        print("Type 'q' to quit")
        print("Type 'c' to clear the screen")
        print("Type 'help' to show this message")

    def run(self) -> None:
        self.clear()
        self.start()

        while self.running:
            try:
                command = input("> ")
            except KeyboardInterrupt:
                break

            match command:
                case "q":
                    break
                case "c":
                    self.clear()
                    continue
                case "help":
                    self.help()
                    continue
                case _:
                    pass

            analyzed_texts = self.morphological_analyzer.analyze(command)
            formatted_texts = self.texts_formatter.format(analyzed_texts)
            print(formatted_texts)

        self.stop()

    @staticmethod
    def clear() -> None:
        # Windows
        if os.name == "nt":
            os.system("cls")
            return
        # macOS
        os.system("clear")


if __name__ == "__main__":
    # NOTE: If you are using `nltk` for the first time, please execute the following code.
    # MorphologicalAnalyzer.setup()

    analyzer = MorphologicalAnalyzer(tag_type="japanese")
    formatter = TextsFormatter(
        # is_list=True,
        is_translate=True,
        language_code="ja",
        color_enabled=True,
        colors_to_remove=["black", "grey", "dark_grey"],
    )
    # formatter = TextsFormatterDeepL(
    #     is_list=True,
    #     api_key=os.environ["DEEPL_API_KEY"],
    #     is_translate=True,
    #     language_code="ja",
    #     color_enabled=True,
    #     colors_to_remove=["black", "grey", "dark_grey"],
    # )

    text = "The quick brown fox jumps over the lazy dog."
    analyzed_text = analyzer.analyze(text)
    formatted_text = formatter.format(analyzed_text)
    print(formatted_text)

    # Use in interactive mode
    prompt = InteractiveMode(morphological_analyzer=analyzer, texts_formatter=formatter)
    prompt.run()
