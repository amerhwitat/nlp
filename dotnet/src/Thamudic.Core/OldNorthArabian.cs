using System.Globalization;
using System.Text;
namespace Chimera.Thamudic;

public sealed record OldNorthArabianCharacter(int CodePoint, string Character, string Name, string Transliteration, string Utf8Hex);

public static class OldNorthArabian
{
    public const int First = 0x10A80;
    public const int Last = 0x10A9F;
    public static readonly string[] VariantForms = { "Dadanitic", "Safaitic", "Hismaic", "Taymanitic", "Minaic", "Thamudic B" };
    public static readonly IReadOnlyList<OldNorthArabianCharacter> Alphabet = Enumerable.Range(First, Last - First + 1)
        .Select(cp => new OldNorthArabianCharacter(cp, char.ConvertFromUtf32(cp), UnicodeName(cp), Transliteration(cp), Utf8Hex(cp)))
        .ToArray();

    public static bool IsOldNorthArabian(int codePoint) => codePoint >= First && codePoint <= Last;
    public static byte[] ToUtf8(int codePoint) => Encoding.UTF8.GetBytes(char.ConvertFromUtf32(codePoint));
    public static string Utf8Hex(int codePoint) => string.Join(" ", ToUtf8(codePoint).Select(b => b.ToString("X2", CultureInfo.InvariantCulture)));
    public static string Transliterate(int codePoint) => codePoint switch
    {
        0x10A80 => "h", 0x10A81 => "l", 0x10A82 => "ḥ", 0x10A83 => "m", 0x10A84 => "q", 0x10A85 => "w", 0x10A86 => "s2", 0x10A87 => "r",
        0x10A88 => "b", 0x10A89 => "t", 0x10A8A => "s1", 0x10A8B => "k", 0x10A8C => "n", 0x10A8D => "ḫ", 0x10A8E => "ṣ", 0x10A8F => "s3",
        0x10A90 => "f", 0x10A91 => "ʼ", 0x10A92 => "ʽ", 0x10A93 => "ḍ", 0x10A94 => "g", 0x10A95 => "d", 0x10A96 => "ġ", 0x10A97 => "ṭ",
        0x10A98 => "z", 0x10A99 => "ḏ", 0x10A9A => "y", 0x10A9B => "ṯ", 0x10A9C => "ẓ", 0x10A9D => "1", 0x10A9E => "10", 0x10A9F => "20", _ => ""
    };
    public static string Transliterate(string text) => string.Concat(text.Select(ch => IsOldNorthArabian(ch) ? Transliterate(ch) : ch.ToString()));
    private static string UnicodeName(int cp) => cp switch
    {
        <= 0x10A9C => $"OLD NORTH ARABIAN LETTER {new[]{"HEH","LAM","HAH","MEEM","QAF","WAW","ES-2","REH","BEH","TEH","ES-1","KAF","NOON","KHAH","SAD","ES-3","FEH","ALEF","AIN","DAD","GEEM","DAL","GHAIN","TAH","ZAIN","THAL","YEH","THEH","ZAH"}[cp-First]}",
        0x10A9D => "OLD NORTH ARABIAN NUMBER ONE", 0x10A9E => "OLD NORTH ARABIAN NUMBER TEN", 0x10A9F => "OLD NORTH ARABIAN NUMBER TWENTY", _ => ""
    };
}
