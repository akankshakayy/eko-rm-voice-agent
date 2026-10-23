package co.eko.sahayak.recsync

import android.content.ContentResolver
import android.net.Uri
import okhttp3.MediaType
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okio.BufferedSink
import okio.source
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

/** The two Sahayak endpoints the app uses: /api/uploader/ping and /api/uploader/recordings. */
class Api(private val serverUrl: String, private val token: String) {

    sealed class Upload {
        data class Ok(val id: Long, val duplicate: Boolean) : Upload()
        /** Bad token: stop the whole run, nothing will succeed until the token is fixed. */
        data class Unauthorized(val detail: String) : Upload()
        /** 400 / 413: this file will never be accepted; skip it for good. */
        data class Rejected(val code: Int, val detail: String) : Upload()
        /** Network or 5xx: try again on the next run. */
        data class Retry(val detail: String) : Upload()
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(20, TimeUnit.SECONDS).writeTimeout(5, TimeUnit.MINUTES).readTimeout(60, TimeUnit.SECONDS)
        .build()

    /** Whose phone this is. A personal token names one RM (locked); a shared token lists the RMs to choose from. */
    data class Ping(val rmId: Int, val rmName: String, val locked: Boolean, val rms: List<Pair<Int, String>>)

    /** Throws IOException with a readable message when the server or token isn't right. */
    fun ping(): Ping {
        val req = Request.Builder().url("$serverUrl/api/uploader/ping").header("Authorization", "Bearer $token").build()
        client.newCall(req).execute().use { r ->
            val body = r.body?.string().orEmpty()
            if (r.code == 401) throw IOException("Token rejected: ${detail(body)}")
            if (!r.isSuccessful) throw IOException("Server said ${r.code}: ${detail(body)}")
            val j = JSONObject(body)
            val list = j.optJSONArray("rms")
            val rms = (0 until (list?.length() ?: 0)).map { list!!.getJSONObject(it).let { o -> o.getInt("id") to o.getString("name") } }
            return Ping(j.optInt("rm_id", -1).takeIf { !j.isNull("rm_id") } ?: -1, j.optString("rm_name", "")
                .takeIf { !j.isNull("rm_name") } ?: "", j.optBoolean("rm_locked", true), rms)
        }
    }

    fun upload(resolver: ContentResolver, file: RecFile, call: CallEntry, deviceId: String, rmId: Int): Upload {
        val audio = object : RequestBody() {
            override fun contentType(): MediaType = mimeFor(file.name).toMediaType()
            override fun contentLength() = file.size
            override fun writeTo(sink: BufferedSink) {
                (resolver.openInputStream(Uri.parse(file.uri)) ?: throw IOException("Can't open ${file.name}"))
                    .source().use { sink.writeAll(it) }
            }
        }
        val form = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("audio", file.name, audio)
            .addFormDataPart("started_at", call.startMs.toString())
            .addFormDataPart("duration_sec", call.durationSec.toString())
            .addFormDataPart("direction", call.direction)
            .addFormDataPart("other_number", call.number)
            .addFormDataPart("device_id", deviceId)
            .addFormDataPart("rm_id", rmId.toString())            // the RM chosen in setup: who made this call
            .build()
        val req = Request.Builder().url("$serverUrl/api/uploader/recordings")
            .header("Authorization", "Bearer $token").post(form).build()
        return try {
            client.newCall(req).execute().use { r ->
                val body = r.body?.string().orEmpty()
                when {
                    r.isSuccessful -> JSONObject(body).let { Upload.Ok(it.optLong("id"), it.optBoolean("duplicate")) }
                    r.code == 401 -> Upload.Unauthorized(detail(body))
                    r.code == 400 || r.code == 413 -> Upload.Rejected(r.code, detail(body))
                    else -> Upload.Retry("Server said ${r.code}: ${detail(body)}")
                }
            }
        } catch (e: IOException) {
            Upload.Retry(e.message ?: e.javaClass.simpleName)
        }
    }

    private fun detail(body: String) = try {
        JSONObject(body).optString("detail", body.take(200))
    } catch (e: Exception) {
        body.take(200)
    }

    companion object {
        const val MAX_BYTES = 100L * 1024 * 1024    // the server answers 413 above this; don't send it at all

        fun mimeFor(name: String) = when (name.substringAfterLast('.', "").lowercase()) {
            "m4a" -> "audio/mp4"
            "mp3" -> "audio/mpeg"
            "aac" -> "audio/aac"
            "amr" -> "audio/amr"
            "3gp" -> "audio/3gpp"
            "ogg", "opus" -> "audio/ogg"
            "wav" -> "audio/wav"
            else -> "application/octet-stream"
        }
    }
}
