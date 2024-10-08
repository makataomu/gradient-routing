# %%
import json
import os
import random
import re
import tempfile

import pytest
import torch as t
from transformers import AutoTokenizer

import factored_representations.string_utils as string_utils
import factored_representations.utils as utils

test_text = """
The unanimous Declaration of the thirteen united States of America,

When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation.

We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness.— That to secure these rights, Governments are instituted among Men, deriving their just powers from the consent of the governed, — That whenever any Form of Government becomes destructive of these ends, it is the Right of the People to alter or to abolish it, and to institute new Government, laying its foundation on such principles and organizing its powers in such form, as to them shall seem most likely to effect their Safety and Happiness. Prudence, indeed, will dictate that Governments long established should not be changed for light and transient causes; and accordingly all experience hath shewn, that mankind are more disposed to suffer, while evils are sufferable, than to right themselves by abolishing the forms to which they are accustomed. But when a long train of abuses and usurpations, pursuing invariably the same Object evinces a design to reduce them under absolute Despotism, it is their right, it is their duty, to throw off such Government, and to provide new Guards for their future security.— Such has been the patient sufferance of these Colonies; and such is now the necessity which constrains them to alter their former Systems of Government. The history of the present King of Great Britain is a history of repeated injuries and usurpations, all having in direct object the establishment of an absolute Tyranny over these States. To prove this, let Facts be submitted to a candid world.
He has refused his Assent to Laws, the most wholesome and necessary for the public good.
He has forbidden his Governors to pass Laws of immediate and pressing importance, unless suspended in their operation till his Assent should be obtained; and when so suspended, he has utterly neglected to attend to them.
He has refused to pass other Laws for the accommodation of large districts of people, unless those people would relinquish the right of Representation in the Legislature, a right inestimable to them and formidable to tyrants only.
He has called together legislative bodies at places unusual, uncomfortable, and distant from the depository of their public Records, for the sole purpose of fatiguing them into compliance with his measures.
He has dissolved Representative Houses repeatedly, for opposing with manly firmness his invasions on the rights of the people.
He has refused for a long time, after such dissolutions, to cause others to be elected; whereby the Legislative powers, incapable of Annihilation, have returned to the People at large for their exercise; the State remaining in the mean time exposed to all the dangers of invasion from without, and convulsions within.
He has endeavoured to prevent the population of these States; for that purpose obstructing the Laws for Naturalization of Foreigners; refusing to pass others to encourage their migrations hither, and raising the conditions of new Appropriations of Lands.
He has obstructed the Administration of Justice, by refusing his Assent to Laws for establishing Judiciary powers.
He has made Judges dependent on his Will alone, for the tenure of their offices, and the amount and payment of their salaries.
He has erected a multitude of New Offices, and sent hither swarms of Officers to harass our people, and eat out their substance.
He has kept among us, in times of peace, Standing Armies without the Consent of our legislatures.
He has affected to render the Military independent of and superior to the Civil power.
He has combined with others to subject us to a jurisdiction foreign to our constitution, and unacknowledged by our laws; giving his Assent to their Acts of pretended Legislation:
For Quartering large bodies of armed troops among us:
For protecting them, by a mock Trial, from punishment for any Murders which they should commit on the Inhabitants of these States:
For cutting off our Trade with all parts of the world:
For imposing Taxes on us without our Consent:
For depriving us in many cases, of the benefits of Trial by Jury:
For transporting us beyond Seas to be tried for pretended offences
For abolishing the free System of English Laws in a neighbouring Province, establishing therein an Arbitrary government, and enlarging its Boundaries so as to render it at once an example and fit instrument for introducing the same absolute rule into these Colonies:
For taking away our Charters, abolishing our most valuable Laws, and altering fundamentally the Forms of our Governments:
For suspending our own Legislatures, and declaring themselves invested with power to legislate for us in all cases whatsoever.
He has abdicated Government here, by declaring us out of his Protection and waging War against us.
He has plundered our seas, ravaged our Coasts, burnt our towns, and destroyed the lives of our people.
He is at this time transporting large Armies of foreign Mercenaries to complete the works of death, desolation and tyranny, already begun with circumstances of Cruelty & perfidy scarcely paralleled in the most barbarous ages, and totally unworthy of the Head of a civilized nation.
He has constrained our fellow Citizens taken Captive on the high Seas to bear Arms against their Country, to become the executioners of their friends and Brethren, or to fall themselves by their Hands.
He has excited domestic insurrections amongst us, and has endeavoured to bring on the inhabitants of our frontiers, the merciless Indian Savages, whose known rule of warfare, is an undistinguished destruction of all ages, sexes and conditions.

In every stage of these Oppressions We have Petitioned for Redress in the most humble terms: Our repeated Petitions have been answered only by repeated injury. A Prince whose character is thus marked by every act which may define a Tyrant, is unfit to be the ruler of a free people.

Nor have We been wanting in attentions to our British brethren. We have warned them from time to time of attempts by their legislature to extend an unwarrantable jurisdiction over us. We have reminded them of the circumstances of our emigration and settlement here. We have appealed to their native justice and magnanimity, and we have conjured them by the ties of our common kindred to disavow these usurpations, which, would inevitably interrupt our connections and correspondence. They too have been deaf to the voice of justice and of consanguinity. We must, therefore, acquiesce in the necessity, which denounces our Separation, and hold them, as we hold the rest of mankind, Enemies in War, in Peace Friends.

We, therefore, the Representatives of the united States of America, in General Congress, Assembled, appealing to the Supreme Judge of the world for the rectitude of our intentions, do, in the Name, and by Authority of the good People of these Colonies, solemnly publish and declare, That these United Colonies are, and of Right ought to be Free and Independent States; that they are Absolved from all Allegiance to the British Crown, and that all political connection between them and the State of Great Britain, is and ought to be totally dissolved; and that as Free and Independent States, they have full Power to levy War, conclude Peace, contract Alliances, establish Commerce, and to do all other Acts and Things which Independent States may of right do. And for the support of this Declaration, with a firm reliance on the protection of divine Providence, we mutually pledge to each other our Lives, our Fortunes and our sacred Honor.
"""


def test_count_target_words_in_story():
    def test_case(story, target_words, expected):
        assert string_utils.count_target_words_in_story(story, target_words) == expected

    test_case("A rainbow after rain.", ["rain"], 1)
    test_case("rain", ["rain"], 1)
    test_case(" rain", ["rain"], 1)
    test_case("rain.", ["rain"], 1)
    test_case("rain,", ["rain"], 1)
    test_case(" rain.", ["rain"], 1)
    test_case(" rain\n", ["rain"], 1)
    test_case("rain", ["rain", "in"], 1)
    test_case("rain", ["in"], 0)
    test_case(" fires", [" fire", " fires"], 1)
    test_case("", ["a"], 0)
    test_case("rain rain rain rain", ["rain"], 4)


def test_encode_decode():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    text = "The quick brown fox jumps over the lazy dog."
    encoded = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    assert text == decoded

    encoded = tokenizer.encode(test_text, add_special_tokens=False)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    assert test_text == decoded


@pytest.fixture
def stories_mixed_len():
    random.seed(42)
    story_len = 200  # pick large enough so there are max_tokens tokens in each story
    stories_from_text = []
    i = 0
    while i < len(test_text) - story_len:
        len_for_this_story = story_len + random.randint(-30, 30)
        story = test_text[i : i + len_for_this_story]
        stories_from_text.append(story)
        i += len_for_this_story
    return stories_from_text


def test_truncated_stories(tokenizer, stories_mixed_len):
    max_tokens = 10

    truncated_stories = string_utils.truncate_stories(
        stories_mixed_len, tokenizer, max_tokens=max_tokens
    )
    list_of_truncated_tokens = [
        tokenizer.encode(story, add_special_tokens=False) for story in truncated_stories
    ]

    for truncated_tokens in list_of_truncated_tokens:
        assert len(truncated_tokens) == max_tokens

    for truncated_tokens, story in zip(list_of_truncated_tokens, stories_mixed_len):
        decoded_str = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        assert story.startswith(decoded_str), f"{story}->\n{decoded_str}"


def test_load_split_and_truncated_dataset(tokenizer):
    stories = string_utils.load_dataset_with_split(
        dataset_name="delphi-suite/stories", split="validation", max_stories=1000
    )

    truncated_stories = string_utils.truncate_stories(stories, tokenizer, max_tokens=30)

    forget_stories, retain_stories = string_utils.split_stories_by_concept(
        truncated_stories, ["cat"]
    )

    def contains_cat(text):
        pattern = r" cat[^a-zA-Z]| cat$"
        match = re.search(pattern, text, re.IGNORECASE)
        return match is not None

    assert len(forget_stories) > 0
    assert len(retain_stories) > 0
    for story in forget_stories:
        assert contains_cat(story), f"String ' cat' not in {story=}."
    for story in retain_stories:
        assert not contains_cat(story), f"String ' cat' in {story=}."


def test_approximate_truncation(tokenizer, stories_mixed_len):
    exact_truncation = string_utils.truncate_stories(
        stories=stories_mixed_len,
        tokenizer=tokenizer,
        max_tokens=30,
    )

    approx_truncation = string_utils.truncate_stories_approximate(
        stories=stories_mixed_len,
        tokenizer=tokenizer,
        max_tokens=30,
        num_sample_stories=100,
        num_characters_buffer=0,
    )

    for truncated_exact, truncated_approx in zip(exact_truncation, approx_truncation):
        min_len = min(len(truncated_exact), len(truncated_approx))
        assert (
            truncated_exact[:min_len] == truncated_approx[:min_len]
        ), f"\n {truncated_exact=}\n{truncated_approx=}"


@pytest.mark.parametrize(
    "fraction, expected_concept, expected_other",
    [
        (0.0, 0, 10),
        (0.1, 1, 9),
        (0.25, 2, 8),
        (0.5, 5, 5),
        (0.75, 7, 3),
        (0.9, 9, 1),
        (1.0, 10, 0),
    ],
)
def test_concat_with_porportion_fractions(fraction, expected_concept, expected_other):
    concept_data = ["A", "B", "C", "D", "E"]
    other_data = ["V", "W", "X", "Y", "Z"]
    result = string_utils.concat_with_porportion(concept_data, other_data, fraction, 10)

    assert len(result) == 10
    assert sum(1 for item in result if item in concept_data) == expected_concept
    assert sum(1 for item in result if item in other_data) == expected_other


def test_concat_with_porportion_large_numbers():
    concept_data = ["A", "B", "C"]
    other_data = ["X", "Y", "Z"]

    # Test with large numbers
    result = string_utils.concat_with_porportion(concept_data, other_data, 0.5, 1000)
    assert len(result) == 1000
    assert sum(1 for item in result if item in concept_data) == 500
    assert sum(1 for item in result if item in other_data) == 500


def test_tokenize_batch_comprehensive(tokenizer):
    # Test data
    short_text = "Hello world"
    long_text = "This is a very long sentence that should definitely be truncated by our function. It should be truncated"
    empty_text = ""
    batch = [short_text, long_text, empty_text]

    # Test parameters
    prepend_bos = True
    truncate_at = 15

    # Call the function
    input_ids, attention_mask = string_utils.tokenize_batch(
        batch, tokenizer, prepend_bos, truncate_at, "right", t.device("cpu")
    )

    # Basic shape checks
    assert isinstance(input_ids, t.Tensor), "input_ids should be a torch.Tensor"
    assert isinstance(
        attention_mask, t.Tensor
    ), "attention_mask should be a torch.Tensor"
    assert (
        input_ids.shape == attention_mask.shape
    ), "input_ids and attention_mask should have the same shape"
    assert input_ids.shape[0] == 3, f"Expected 3 sequences, got {input_ids.shape[0]}"
    assert (
        input_ids.shape[1] <= truncate_at
    ), f"Sequence length {input_ids.shape[1]} exceeds truncate_at {truncate_at}"

    # Check BOS token prepending
    assert (
        input_ids[:, 0] == tokenizer.bos_token_id
    ).all(), "BOS token should be prepended to all sequences"

    # Check truncation
    assert (
        input_ids.shape[1] == truncate_at
    ), f"Expected length {truncate_at}, got {input_ids.shape[1]}"

    # Check padding
    pad_token_id = tokenizer.pad_token_id
    assert input_ids[0][-1] == pad_token_id, "Short sequence should be padded"
    assert input_ids[2][-1] == pad_token_id, "Empty sequence should be padded"

    # Check attention mask
    assert (
        attention_mask[0].sum() > attention_mask[2].sum()
    ), "Short text should have more attention tokens than empty text"
    assert (
        attention_mask[1].sum() == truncate_at
    ), "Long text should have full attention up to truncation point"


def test_tokenize_batch_padding_side(tokenizer):
    prompts = [
        "This is a short prompt",
        "This is a much longer and more elaborate prompt that is meant to test the padding function",
    ]
    device = "cpu"
    short_prompt_tokens = tokenizer(prompts[0], return_tensors="pt")["input_ids"]
    long_prompt_tokens = tokenizer(prompts[1], return_tensors="pt")["input_ids"]
    short_prompt_token_len = short_prompt_tokens.size(-1)
    long_prompt_token_len = long_prompt_tokens.size(-1)
    token_len_diff = long_prompt_token_len - short_prompt_token_len

    _, attn_mask = string_utils.tokenize_batch(
        prompts,
        tokenizer,
        prepend_bos=False,
        truncate_at=9999,
        padding_side="left",
        device=t.device(device),
    )
    assert isinstance(attn_mask, t.Tensor)
    assert attn_mask.shape == (2, long_prompt_token_len)

    padding_zeros = t.zeros(token_len_diff, dtype=t.long, device=device)
    short_prompt_ones = t.ones(short_prompt_token_len, dtype=t.long, device=device)

    short_prompt_attn_mask = attn_mask[0]
    assert t.allclose(short_prompt_attn_mask[:token_len_diff], padding_zeros)
    assert t.allclose(short_prompt_attn_mask[token_len_diff:], short_prompt_ones)

    long_prompt_attention_mask = attn_mask[1]
    long_prompt_ones = t.ones(long_prompt_token_len, dtype=t.long, device=device)
    assert t.allclose(long_prompt_attention_mask, long_prompt_ones)

    _, attn_mask = string_utils.tokenize_batch(
        prompts,
        tokenizer,
        prepend_bos=False,
        truncate_at=9999,
        padding_side="right",
        device=t.device(device),
    )
    assert isinstance(attn_mask, t.Tensor)
    assert attn_mask.shape == (2, long_prompt_token_len)

    short_prompt_attn_mask = attn_mask[0]
    assert t.allclose(
        short_prompt_attn_mask[:short_prompt_token_len], short_prompt_ones
    )
    assert t.allclose(short_prompt_attn_mask[short_prompt_token_len:], padding_zeros)

    long_prompt_attention_mask = attn_mask[1]
    assert t.allclose(long_prompt_attention_mask, long_prompt_ones)


def test_get_first_row_and_delete():
    # Create a temporary JSONL file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
        tmp_filename = tmp_file.name
        data: list[dict] = [
            {
                "layers_to_mask": [0, 1, 2],
                "to_expand": {"d_mlp": 64, "n_heads": 2},
                "masking_scheme": "full_seq",
                "masking_type": "ddbp",
            },
            {"text": "This is the first row."},
            {"text": "This is the second row."},
            {"text": "This is the third row."},
        ]
        for row in data:
            tmp_file.write(json.dumps(row) + "\n")

    # Test getting the first row and deleting it
    first_row = utils.get_first_row_and_delete(tmp_filename)
    assert first_row == data[0]

    # Check that the first row has been deleted
    with open(tmp_filename, "r") as f:
        remaining_data = [json.loads(line.strip()) for line in f.readlines()]
    assert remaining_data == data[1:]

    # Clean up the temporary file
    os.remove(tmp_filename)

    # Test with an empty file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
        tmp_filename = tmp_file.name
    first_row = utils.get_first_row_and_delete(tmp_filename)
    assert first_row is None

    # Test with a file containing an empty first line
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
        tmp_filename = tmp_file.name
        tmp_file.write("\n")
        tmp_file.write(json.dumps(data[1]) + "\n")
    first_row = utils.get_first_row_and_delete(tmp_filename)
    assert first_row is None

    # Clean up the temporary files
    os.remove(tmp_filename)
