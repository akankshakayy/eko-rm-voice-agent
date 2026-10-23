package co.eko.sahayak.recsync

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKeys
import java.util.UUID

/** Setup values (token kept in EncryptedSharedPreferences), run status, and the handled-files ledger. */
class Settings(context: Context) {
    private val app = context.applicationContext

    private val secure: SharedPreferences = EncryptedSharedPreferences.create(
        "secure", MasterKeys.getOrCreate(MasterKeys.AES256_GCM_SPEC), app,
        EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
        EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM)
    private val prefs = app.getSharedPreferences("settings", Context.MODE_PRIVATE)
    private val handled = app.getSharedPreferences("handled", Context.MODE_PRIVATE)   // file key -> mtime

    var serverUrl: String
        get() = prefs.getString("server_url", "") ?: ""
        set(v) = prefs.edit().putString("server_url", v.trim().trimEnd('/')).apply()

    var token: String
        get() = secure.getString("token", "") ?: ""
        set(v) = secure.edit().putString("token", v.trim()).apply()

    /** The RM this phone uploads for: picked in setup, or fixed by a personal token (rmLocked). */
    var rmId: Int
        get() = prefs.getInt("rm_id", -1)
        set(v) = prefs.edit().putInt("rm_id", v).apply()
    var rmName: String
        get() = prefs.getString("rm_name", "") ?: ""
        set(v) = prefs.edit().putString("rm_name", v).apply()
    var rmLocked: Boolean
        get() = prefs.getBoolean("rm_locked", false)
        set(v) = prefs.edit().putBoolean("rm_locked", v).apply()

    /** Subscription id of the Eko SIM, plus every other id the call log may use for it (see Matcher.isEkoSim). */
    var ekoSubId: Int
        get() = prefs.getInt("eko_sub_id", -1)
        set(v) = prefs.edit().putInt("eko_sub_id", v).apply()
    var ekoSimLabel: String
        get() = prefs.getString("eko_sim_label", "") ?: ""
        set(v) = prefs.edit().putString("eko_sim_label", v).apply()
    var ekoAccountIds: Set<String>
        get() = prefs.getStringSet("eko_account_ids", emptySet()) ?: emptySet()
        set(v) = prefs.edit().putStringSet("eko_account_ids", v).apply()

    var folderUri: String
        get() = prefs.getString("folder_uri", "") ?: ""
        set(v) = prefs.edit().putString("folder_uri", v).apply()

    var wifiOnly: Boolean
        get() = prefs.getBoolean("wifi_only", false)
        set(v) = prefs.edit().putBoolean("wifi_only", v).apply()

    val deviceId: String
        get() = prefs.getString("device_id", null) ?: UUID.randomUUID().toString().also {
            prefs.edit().putString("device_id", it).apply()
        }

    /** Files last-modified at or before this were all handled. First run looks back BACKFILL_DAYS. */
    var watermarkMs: Long
        get() = prefs.getLong("watermark", System.currentTimeMillis() - BACKFILL_DAYS * DAY_MS)
        set(v) = prefs.edit().putLong("watermark", v).apply()

    fun isHandled(f: RecFile) = handled.contains(f.key)
    fun markHandled(f: RecFile) = handled.edit().putLong(f.key, f.mtimeMs).apply()

    /** Forget ledger entries well behind the watermark; those files are never listed again. */
    fun pruneHandled() {
        val cutoff = watermarkMs - OVERLAP_MS - DAY_MS
        val e = handled.edit()
        handled.all.forEach { (k, v) -> if ((v as? Long ?: 0L) < cutoff) e.remove(k) }
        e.apply()
    }

    val isConfigured get() = serverUrl.isNotEmpty() && token.isNotEmpty() && rmId > 0 && ekoAccountIds.isNotEmpty() &&
        folderUri.isNotEmpty()

    // ---- last run status
    fun saveStatus(s: RunStatus) = prefs.edit()
        .putLong("last_run", s.at).putInt("uploaded", s.uploaded).putInt("duplicates", s.duplicates)
        .putInt("skipped", s.skipped).putInt("failed", s.failed).putString("last_error", s.error).apply()

    fun status() = RunStatus(prefs.getLong("last_run", 0), prefs.getInt("uploaded", 0),
        prefs.getInt("duplicates", 0), prefs.getInt("skipped", 0), prefs.getInt("failed", 0),
        prefs.getString("last_error", null))

    companion object {
        const val DAY_MS = 24 * 60 * 60 * 1000L
        const val BACKFILL_DAYS = 7L
        /** Re-list files this far behind the watermark, in case a recorder sets mtime late. */
        const val OVERLAP_MS = 6 * 60 * 60 * 1000L
    }
}

data class RunStatus(val at: Long, val uploaded: Int, val duplicates: Int, val skipped: Int, val failed: Int,
                     val error: String?)
