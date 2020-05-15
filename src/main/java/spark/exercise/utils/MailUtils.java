package spark.exercise.utils;

import javax.mail.BodyPart;
import javax.mail.MessagingException;
import javax.mail.Session;
import javax.mail.internet.MimeMessage;
import javax.mail.internet.MimeMultipart;
import java.io.*;
import java.util.*;

public class MailUtils {

    public static class MailMsg implements Serializable {
		private static final long serialVersionUID = 432702124074913722L;
		public final String id;
        public final String text;

        public MailMsg(String id, String text) {
            this.id = id;
            this.text = text;
        }
    }

    private static final Session MAIL_SESSION = Session.getDefaultInstance(new Properties());
    
    public static String getMailBodyText(String pathToMessage) {

        try (InputStream is = new FileInputStream(pathToMessage)) {
            MimeMessage message = new MimeMessage(MAIL_SESSION, is);

            if (message.isMimeType("multipart/*")) {
                String text = "";
                MimeMultipart content = (MimeMultipart) message.getContent();
                for (int p = 0; p < content.getCount(); p++) {
                    BodyPart part = content.getBodyPart(p);
                    if (part.isMimeType("text/*")) {
                        text += part.getContent().toString();
                    }
                }
                return text;
            }
            
            if (message.isMimeType("text/*")) {
                return message.getContent().toString();
            }

        } catch(IOException | MessagingException exc) {
	        //NOTE: for simplicity reasons we don't do error handling
	        	//and instead silently ignore reading/parsing errors
	        	//exc.printStackTrace();
        }

        return null;
    }

}
