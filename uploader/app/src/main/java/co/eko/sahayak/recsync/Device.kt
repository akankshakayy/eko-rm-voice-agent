package co.eko.sahayak.recsync

import android.annotation.SuppressLint
import android.content.Context
import android.net.Uri
import android.os.Build
import android.provider.CallLog
import android.provider.DocumentsContract
import android.telecom.TelecomManager
import android.telephony.SubscriptionManager
import android.telephony.TelephonyManager

/** Everything that reads the phone: SIMs, the call log, and the recordings folder (via SAF). */
object Device {

    data class Sim(val subId: Int, val label: String, val accountIds: Set<String>)

    /**
     * Active SIMs with every id the call log might use for each (see Matcher.isEkoSim):
     * the subscription id, the ICCID (readable only up to Android 9 for normal apps), and the Telecom
     * PhoneAccountHandle id, which is exactly what CallLog.PHONE_ACCOUNT_ID stores.
     */
    @SuppressLint("MissingPermission")   // READ_PHONE_STATE is checked by the caller
    fun sims(context: Context): List<Sim> {
        val sm = context.getSystemService(SubscriptionManager::class.java)
        val tm = context.getSystemService(TelephonyManager::class.java)
        val telecom = context.getSystemService(TelecomManager::class.java)
        val subs = sm.activeSubscriptionInfoList.orEmpty()
        val handles = try { telecom.callCapablePhoneAccounts } catch (e: SecurityException) { emptyList() }
        return subs.map { info ->
            val ids = mutableSetOf(info.subscriptionId.toString())
            info.iccId?.takeIf { it.isNotBlank() }?.let(ids::add)
            for (h in handles) {
                val mapped = if (Build.VERSION.SDK_INT >= 30) {
                    tm.getSubscriptionId(h) == info.subscriptionId
                } else {
                    // Before Android 11 there's no handle -> subscription lookup; match by id or label
                    h.id == info.subscriptionId.toString() || (info.iccId?.let { h.id.startsWith(it) } ?: false) ||
                        telecom.getPhoneAccount(h)?.label?.toString() == info.displayName?.toString()
                }
                if (mapped) ids.add(h.id)
            }
            val number = if (Build.VERSION.SDK_INT >= 33) "" else (info.number ?: "")
            Sim(info.subscriptionId,
                "SIM ${info.simSlotIndex + 1}: ${info.displayName ?: info.carrierName ?: "?"} ${number}".trim(), ids)
        }
    }

    /** Answered calls since [sinceMs]. */
    @SuppressLint("MissingPermission")   // READ_CALL_LOG is checked by the caller
    fun calls(context: Context, sinceMs: Long): List<CallEntry> {
        val cols = arrayOf(CallLog.Calls.DATE, CallLog.Calls.DURATION, CallLog.Calls.NUMBER,
            CallLog.Calls.PHONE_ACCOUNT_ID, CallLog.Calls.TYPE)
        val out = mutableListOf<CallEntry>()
        context.contentResolver.query(CallLog.Calls.CONTENT_URI, cols, "${CallLog.Calls.DATE} >= ?",
            arrayOf(sinceMs.toString()), "${CallLog.Calls.DATE} ASC")?.use { c ->
            while (c.moveToNext()) {
                val direction = when (c.getInt(4)) {
                    CallLog.Calls.INCOMING_TYPE, CallLog.Calls.ANSWERED_EXTERNALLY_TYPE -> "incoming"
                    CallLog.Calls.OUTGOING_TYPE -> "outgoing"
                    else -> "unknown"
                }
                out.add(CallEntry(c.getLong(0), c.getLong(1), c.getString(2) ?: "", c.getString(3), direction))
            }
        }
        return out
    }

    /** Audio files under the picked folder (and up to 3 levels of subfolders) modified after [sinceMs]. */
    fun recordings(context: Context, treeUri: Uri, sinceMs: Long): List<RecFile> {
        val out = mutableListOf<RecFile>()
        fun walk(docId: String, depth: Int) {
            val children = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, docId)
            val cols = arrayOf(DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                DocumentsContract.Document.COLUMN_DISPLAY_NAME, DocumentsContract.Document.COLUMN_SIZE,
                DocumentsContract.Document.COLUMN_LAST_MODIFIED, DocumentsContract.Document.COLUMN_MIME_TYPE)
            context.contentResolver.query(children, cols, null, null, null)?.use { c ->
                while (c.moveToNext()) {
                    val id = c.getString(0)
                    val name = c.getString(1) ?: continue
                    if (c.getString(4) == DocumentsContract.Document.MIME_TYPE_DIR) {
                        if (depth < 3) walk(id, depth + 1)
                        continue
                    }
                    val mtime = c.getLong(3)
                    if (mtime > sinceMs && Matcher.isAudio(name)) {
                        val uri = DocumentsContract.buildDocumentUriUsingTree(treeUri, id)
                        out.add(RecFile(uri.toString(), name, c.getLong(2), mtime))
                    }
                }
            }
        }
        walk(DocumentsContract.getTreeDocumentId(treeUri), 0)
        return out.sortedBy { it.mtimeMs }
    }
}
