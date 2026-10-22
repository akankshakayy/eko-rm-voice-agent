package co.eko.sahayak.recsync

import java.time.LocalDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter

/**
 * Pure matching logic (no Android imports) so it can be unit-tested on the JVM.
 *
 * A dialer recording file doesn't say which call or which SIM it came from, so we line it up with the
 * call log: the file is usually closed (last-modified) when the call ends, i.e. at DATE + DURATION.
 */

data class CallEntry(
    val startMs: Long,          // CallLog.Calls.DATE
    val durationSec: Long,      // CallLog.Calls.DURATION
    val number: String,         // CallLog.Calls.NUMBER
    val accountId: String?,     // CallLog.Calls.PHONE_ACCOUNT_ID (subscription id, ICCID or OEM id)
    val direction: String,      // incoming | outgoing | unknown
) {
    val endMs get() = startMs + durationSec * 1000
}

data class RecFile(val uri: String, val name: String, val size: Long, val mtimeMs: Long) {
    /** Stable identity for "already handled" bookkeeping. */
    val key get() = "$uri|$size|$mtimeMs"
}

object Matcher {
    const val TOLERANCE_MS = 3 * 60 * 1000L
    val AUDIO_EXTENSIONS = setOf("m4a", "mp3", "aac", "amr", "3gp", "ogg", "opus", "wav")

    fun isAudio(name: String) = name.substringAfterLast('.', "").lowercase() in AUDIO_EXTENSIONS

    /** Last 10 digits: enough to compare Indian numbers with or without +91 / 0 prefixes. */
    fun tail(number: String): String = number.filter(Char::isDigit).takeLast(10)

    /** Phone numbers written in the file name, e.g. "Call recording 9876543210_20261021_143012.m4a". */
    fun numbersIn(name: String): Set<String> =
        Regex("""\+?\d[\d\s-]{8,15}\d""").findAll(STAMP.replace(name, " "))
            .map { tail(it.value) }.filter { it.length == 10 }.toSet()

    private val STAMP = Regex("""(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})[-_ T.]?(\d{2})[-_.:]?(\d{2})[-_.:]?(\d{2})""")

    /** A yyyyMMdd_HHmmss timestamp in the file name (local time), which most recorders write at call start. */
    fun timestampIn(name: String, zone: ZoneId = ZoneId.systemDefault()): Long? {
        val m = STAMP.find(name) ?: return null
        val (y, mo, d, h, mi, s) = m.destructured
        return try {
            LocalDateTime.parse("$y$mo$d$h$mi$s", DateTimeFormatter.ofPattern("yyyyMMddHHmmss"))
                .atZone(zone).toInstant().toEpochMilli()
        } catch (e: Exception) {
            null
        }
    }

    /**
     * The call this recording belongs to, or null. Candidates are answered calls whose
     * [start - tolerance, end + tolerance] window contains the file's mtime (or the file-name timestamp).
     * Closest time wins. If the file name contains a phone number, only calls with that number qualify.
     */
    fun match(file: RecFile, calls: List<CallEntry>, zone: ZoneId = ZoneId.systemDefault()): CallEntry? {
        val stamp = timestampIn(file.name, zone)
        val numbers = numbersIn(file.name)
        return calls.asSequence()
            .filter { it.durationSec > 0 }
            .mapNotNull { c ->
                val byMtime = if (file.mtimeMs in (c.startMs - TOLERANCE_MS)..(c.endMs + TOLERANCE_MS))
                    Math.abs(file.mtimeMs - c.endMs) else null
                val byStamp = if (stamp != null && stamp in (c.startMs - TOLERANCE_MS)..(c.endMs + TOLERANCE_MS))
                    Math.abs(stamp - c.startMs) else null
                val dist = listOfNotNull(byMtime, byStamp).minOrNull() ?: return@mapNotNull null
                val numberHit = numbers.isNotEmpty() && tail(c.number) in numbers
                // The file name names a different number: not this call, however close the timing
                if (numbers.isNotEmpty() && !numberHit) return@mapNotNull null
                c to dist
            }
            .sortedBy { it.second }
            .firstOrNull()?.first
    }

    /**
     * Was this call on the Eko SIM? CallLog.PHONE_ACCOUNT_ID is OEM-specific: most phones store the
     * subscription id ("2"), some (older Samsung, some MediaTek) store the SIM's ICCID, sometimes with a
     * trailing 'F' or truncated. [ekoIds] holds every id we know for the Eko SIM: its subscription id,
     * its ICCID when readable, and the Telecom PhoneAccountHandle id(s) mapped to it.
     */
    fun isEkoSim(accountId: String?, ekoIds: Set<String>): Boolean {
        val id = accountId?.trim()?.trimEnd('F', 'f')?.lowercase()
        if (id.isNullOrEmpty()) return false
        return ekoIds.any { raw ->
            val e = raw.trim().trimEnd('F', 'f').lowercase()
            e.isNotEmpty() && (e == id || (minOf(e.length, id.length) >= 18 && (e.startsWith(id) || id.startsWith(e))))
        }
    }

    private val PRIVATE_V4 = listOf(Regex("""^10\.\d+\.\d+\.\d+$"""),
        Regex("""^192\.168\.\d+\.\d+$"""), Regex("""^172\.(1[6-9]|2\d|3[01])\.\d+\.\d+$"""))

    /** https anywhere; plain http only to a private LAN address (for testing against a laptop). */
    fun serverUrlProblem(url: String): String? {
        val m = Regex("""^(https?)://([^/:?#]+)(:\d+)?(/.*)?$""", RegexOption.IGNORE_CASE).find(url.trim())
            ?: return "Enter a full address, e.g. https://sahayak.eko.in"
        val (scheme, host) = m.destructured
        if (scheme.lowercase() == "http" && PRIVATE_V4.none { it.matches(host) })
            return "Use https (plain http is only allowed for a 192.168.x / 10.x test server)"
        return null
    }
}
