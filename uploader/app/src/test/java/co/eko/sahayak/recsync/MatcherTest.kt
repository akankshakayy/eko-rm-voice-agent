package co.eko.sahayak.recsync

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalTime
import java.time.ZoneId
import java.time.ZonedDateTime

class MatcherTest {
    private val ist = ZoneId.of("Asia/Kolkata")
    private fun at(h: Int, m: Int, s: Int = 0) =
        ZonedDateTime.of(2026, 10, 21, h, m, s, 0, ist).toInstant().toEpochMilli()

    private val ekoCall = CallEntry(at(14, 30), 120, "+91 98765 43210", "2", "outgoing")       // ends 14:32
    private val personalCall = CallEntry(at(14, 33), 60, "9123456780", "1", "incoming")       // ends 14:34
    private val missed = CallEntry(at(14, 40), 0, "9988776655", "2", "incoming")

    @Test fun fileClosedAtCallEndMatchesThatCall() {
        val f = RecFile("u1", "Call recording.m4a", 1000, at(14, 32, 3))
        assertEquals(ekoCall, Matcher.match(f, listOf(ekoCall, personalCall), ist))
    }

    @Test fun closestCallWinsWhenWindowsOverlap() {
        val f = RecFile("u2", "rec.m4a", 1000, at(14, 34, 2))
        assertEquals(personalCall, Matcher.match(f, listOf(ekoCall, personalCall), ist))
    }

    @Test fun numberInFileNameOverridesTiming() {
        // Timing alone would pick the personal call; the file name says it's the Eko-SIM call's number
        val f = RecFile("u3", "9876543210_rec.m4a", 1000, at(14, 34, 2))
        assertEquals(ekoCall, Matcher.match(f, listOf(ekoCall, personalCall), ist))
    }

    @Test fun fileNameTimestampIsUsedWhenMtimeIsOff() {
        // Copied file: mtime is hours later, but the name carries the call's start time
        val f = RecFile("u4", "Call_20261021_143001.m4a", 1000, at(18, 0))
        assertEquals(ekoCall, Matcher.match(f, listOf(ekoCall, personalCall), ist))
        assertEquals(at(14, 30, 1), Matcher.timestampIn(f.name, ist))
        assertTrue("timestamp digits aren't a phone number", Matcher.numbersIn(f.name).isEmpty())
    }

    @Test fun noCallNearbyMeansNoMatch() {
        assertNull(Matcher.match(RecFile("u5", "x.m4a", 1, at(9, 0)), listOf(ekoCall, personalCall), ist))
    }

    @Test fun missedCallsAreIgnored() {
        assertNull(Matcher.match(RecFile("u6", "x.m4a", 1, at(14, 40, 5)), listOf(missed), ist))
    }

    @Test fun ekoSimBySubscriptionIdOrIccid() {
        val ids = setOf("2", "8991000900000012345")
        assertTrue(Matcher.isEkoSim("2", ids))
        assertFalse(Matcher.isEkoSim("1", ids))
        assertFalse(Matcher.isEkoSim(null, ids))
        assertFalse(Matcher.isEkoSim("", ids))
        assertTrue(Matcher.isEkoSim("8991000900000012345F", ids))     // trailing F
        assertTrue(Matcher.isEkoSim("899100090000001234", ids))       // truncated ICCID
        assertFalse(Matcher.isEkoSim("8991000900000099999", ids))
        assertFalse("short ids must match exactly", Matcher.isEkoSim("22", ids))
    }

    @Test fun onlyAudioFiles() {
        assertTrue(Matcher.isAudio("a.M4A"))
        assertTrue(Matcher.isAudio("b.amr"))
        assertFalse(Matcher.isAudio("notes.txt"))
        assertFalse(Matcher.isAudio(".nomedia"))
    }

    @Test fun serverUrlRules() {
        assertNull(Matcher.serverUrlProblem("https://sahayak.eko.in"))
        assertNull(Matcher.serverUrlProblem("http://192.168.1.20:8000"))
        assertNull(Matcher.serverUrlProblem("http://10.0.0.5:8000/"))
        assertNull(Matcher.serverUrlProblem("http://172.16.4.2"))
        assertNotNull(Matcher.serverUrlProblem("http://sahayak.eko.in"))
        assertNotNull(Matcher.serverUrlProblem("http://172.32.0.1"))
        assertNotNull(Matcher.serverUrlProblem("sahayak.eko.in"))
    }

    @Test fun dailyRunIsNextEightPm() {
        val before = ZonedDateTime.of(2026, 10, 21, 18, 0, 0, 0, ist)
        assertEquals(2 * 60L, UploadWorker.delayUntilNext(LocalTime.of(20, 0), before).toMinutes())
        val after = ZonedDateTime.of(2026, 10, 21, 21, 0, 0, 0, ist)
        assertEquals(23 * 60L, UploadWorker.delayUntilNext(LocalTime.of(20, 0), after).toMinutes())
    }
}
