package co.eko.sahayak.recsync

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.net.Uri
import android.util.Log
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.Worker
import androidx.work.WorkerParameters
import java.time.Duration
import java.time.LocalDate
import java.time.LocalTime
import java.time.ZoneId
import java.time.ZonedDateTime
import java.util.concurrent.TimeUnit

/**
 * One upload run: list new recordings, keep the ones that line up with an answered call on the Eko SIM,
 * upload them, and skip everything else (calls on the personal SIM are never sent).
 */
class UploadWorker(context: Context, params: WorkerParameters) : Worker(context, params) {

    override fun doWork(): Result {
        val s = Settings(applicationContext)
        var uploaded = 0; var duplicates = 0; var skipped = 0; var failed = 0
        var error: String? = null
        fun finish(result: Result): Result {
            s.saveStatus(RunStatus(System.currentTimeMillis(), uploaded, duplicates, skipped, failed, error))
            return result
        }

        if (!s.isConfigured) { error = "Setup isn't finished"; return finish(Result.failure()) }
        for (p in listOf(Manifest.permission.READ_CALL_LOG, Manifest.permission.READ_PHONE_STATE)) {
            if (applicationContext.checkSelfPermission(p) != PackageManager.PERMISSION_GRANTED) {
                error = "Permission missing: ${p.substringAfterLast('.')}"; return finish(Result.failure())
            }
        }

        val now = System.currentTimeMillis()
        val since = s.watermarkMs - Settings.OVERLAP_MS
        val files = try {
            Device.recordings(applicationContext, Uri.parse(s.folderUri), since)
        } catch (e: Exception) {
            error = "Can't read the recordings folder (pick it again): ${e.message}"; return finish(Result.failure())
        }
        val calls = Device.calls(applicationContext, since - Settings.DAY_MS)
        val api = Api(s.serverUrl, s.token)

        // The watermark only moves past an unbroken run of handled files, so a failure is retried next time
        var watermark = s.watermarkMs
        var blocked = false
        var retry = false
        for (f in files) {
            if (s.isHandled(f)) { if (!blocked) watermark = maxOf(watermark, f.mtimeMs); continue }
            val age = now - f.mtimeMs
            if (age < STILL_WRITING_MS) { blocked = true; continue }

            val call = Matcher.match(f, calls)
            val outcome: String = when {
                f.size > Api.MAX_BYTES -> "skip: over 100 MB"
                call == null && age < CALL_LOG_LAG_MS -> { blocked = true; continue }   // call log may lag
                call == null -> "skip: no matching call"
                !Matcher.isEkoSim(call.accountId, s.ekoAccountIds) -> "skip: not the Eko SIM"
                else -> when (val r = api.upload(applicationContext.contentResolver, f, call, s.deviceId)) {
                    is Api.Upload.Ok -> if (r.duplicate) "duplicate" else "uploaded"
                    is Api.Upload.Rejected -> "rejected ${r.code}: ${r.detail}"
                    is Api.Upload.Unauthorized -> {
                        error = "Token rejected: ${r.detail}. Ask the Sahayak admin for a new token."
                        return finish(Result.failure())
                    }
                    is Api.Upload.Retry -> { error = r.detail; retry = true; break }
                }
            }
            Log.i(TAG, "${f.name}: $outcome")
            when {
                outcome == "uploaded" -> uploaded++
                outcome == "duplicate" -> duplicates++
                outcome.startsWith("skip") -> skipped++
                else -> { failed++; error = "${f.name}: $outcome" }
            }
            s.markHandled(f)
            if (!blocked) watermark = maxOf(watermark, f.mtimeMs)
        }
        if (retry) failed++
        s.watermarkMs = watermark
        s.pruneHandled()
        return finish(if (retry) Result.retry() else Result.success())
    }

    companion object {
        const val TAG = "RecSync"
        private const val DAILY = "daily-upload"
        private const val NOW = "upload-now"
        private const val STILL_WRITING_MS = 2 * 60 * 1000L
        private const val CALL_LOG_LAG_MS = 15 * 60 * 1000L
        private val RUN_AT: LocalTime = LocalTime.of(20, 0)

        private fun constraints(s: Settings) = Constraints.Builder()
            .setRequiredNetworkType(if (s.wifiOnly) NetworkType.UNMETERED else NetworkType.CONNECTED)
            .setRequiresBatteryNotLow(true)
            .build()

        /** Once a day, starting at the next 8 pm (after the RM's working day). */
        fun scheduleDaily(context: Context) {
            val s = Settings(context)
            val req = PeriodicWorkRequestBuilder<UploadWorker>(24, TimeUnit.HOURS)
                .setInitialDelay(delayUntilNext(RUN_AT, ZonedDateTime.now()).toMillis(), TimeUnit.MILLISECONDS)
                .setConstraints(constraints(s))
                .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 15, TimeUnit.MINUTES)
                .build()
            WorkManager.getInstance(context).enqueueUniquePeriodicWork(DAILY, ExistingPeriodicWorkPolicy.UPDATE, req)
        }

        fun runNow(context: Context) {
            val req = OneTimeWorkRequestBuilder<UploadWorker>()
                .setConstraints(Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build())
                .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 1, TimeUnit.MINUTES)
                .build()
            WorkManager.getInstance(context).enqueueUniqueWork(NOW, ExistingWorkPolicy.REPLACE, req)
        }

        fun delayUntilNext(at: LocalTime, now: ZonedDateTime, zone: ZoneId = now.zone): Duration {
            var next = ZonedDateTime.of(LocalDate.from(now), at, zone)
            if (!next.isAfter(now)) next = next.plusDays(1)
            return Duration.between(now, next)
        }
    }
}
